from __future__ import annotations



def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[104, 64, 1, 1]", arg4_1: "f32[104]", arg5_1: "f32[104]", arg6_1: "f32[26, 26, 3, 3]", arg7_1: "f32[26]", arg8_1: "f32[26]", arg9_1: "f32[26, 26, 3, 3]", arg10_1: "f32[26]", arg11_1: "f32[26]", arg12_1: "f32[26, 26, 3, 3]", arg13_1: "f32[26]", arg14_1: "f32[26]", arg15_1: "f32[256, 104, 1, 1]", arg16_1: "f32[256]", arg17_1: "f32[256]", arg18_1: "f32[256, 64, 1, 1]", arg19_1: "f32[256]", arg20_1: "f32[256]", arg21_1: "f32[104, 256, 1, 1]", arg22_1: "f32[104]", arg23_1: "f32[104]", arg24_1: "f32[26, 26, 3, 3]", arg25_1: "f32[26]", arg26_1: "f32[26]", arg27_1: "f32[26, 26, 3, 3]", arg28_1: "f32[26]", arg29_1: "f32[26]", arg30_1: "f32[26, 26, 3, 3]", arg31_1: "f32[26]", arg32_1: "f32[26]", arg33_1: "f32[256, 104, 1, 1]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[104, 256, 1, 1]", arg37_1: "f32[104]", arg38_1: "f32[104]", arg39_1: "f32[26, 26, 3, 3]", arg40_1: "f32[26]", arg41_1: "f32[26]", arg42_1: "f32[26, 26, 3, 3]", arg43_1: "f32[26]", arg44_1: "f32[26]", arg45_1: "f32[26, 26, 3, 3]", arg46_1: "f32[26]", arg47_1: "f32[26]", arg48_1: "f32[256, 104, 1, 1]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[208, 256, 1, 1]", arg52_1: "f32[208]", arg53_1: "f32[208]", arg54_1: "f32[52, 52, 3, 3]", arg55_1: "f32[52]", arg56_1: "f32[52]", arg57_1: "f32[52, 52, 3, 3]", arg58_1: "f32[52]", arg59_1: "f32[52]", arg60_1: "f32[52, 52, 3, 3]", arg61_1: "f32[52]", arg62_1: "f32[52]", arg63_1: "f32[512, 208, 1, 1]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[512, 256, 1, 1]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[208, 512, 1, 1]", arg70_1: "f32[208]", arg71_1: "f32[208]", arg72_1: "f32[52, 52, 3, 3]", arg73_1: "f32[52]", arg74_1: "f32[52]", arg75_1: "f32[52, 52, 3, 3]", arg76_1: "f32[52]", arg77_1: "f32[52]", arg78_1: "f32[52, 52, 3, 3]", arg79_1: "f32[52]", arg80_1: "f32[52]", arg81_1: "f32[512, 208, 1, 1]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[208, 512, 1, 1]", arg85_1: "f32[208]", arg86_1: "f32[208]", arg87_1: "f32[52, 52, 3, 3]", arg88_1: "f32[52]", arg89_1: "f32[52]", arg90_1: "f32[52, 52, 3, 3]", arg91_1: "f32[52]", arg92_1: "f32[52]", arg93_1: "f32[52, 52, 3, 3]", arg94_1: "f32[52]", arg95_1: "f32[52]", arg96_1: "f32[512, 208, 1, 1]", arg97_1: "f32[512]", arg98_1: "f32[512]", arg99_1: "f32[208, 512, 1, 1]", arg100_1: "f32[208]", arg101_1: "f32[208]", arg102_1: "f32[52, 52, 3, 3]", arg103_1: "f32[52]", arg104_1: "f32[52]", arg105_1: "f32[52, 52, 3, 3]", arg106_1: "f32[52]", arg107_1: "f32[52]", arg108_1: "f32[52, 52, 3, 3]", arg109_1: "f32[52]", arg110_1: "f32[52]", arg111_1: "f32[512, 208, 1, 1]", arg112_1: "f32[512]", arg113_1: "f32[512]", arg114_1: "f32[416, 512, 1, 1]", arg115_1: "f32[416]", arg116_1: "f32[416]", arg117_1: "f32[104, 104, 3, 3]", arg118_1: "f32[104]", arg119_1: "f32[104]", arg120_1: "f32[104, 104, 3, 3]", arg121_1: "f32[104]", arg122_1: "f32[104]", arg123_1: "f32[104, 104, 3, 3]", arg124_1: "f32[104]", arg125_1: "f32[104]", arg126_1: "f32[1024, 416, 1, 1]", arg127_1: "f32[1024]", arg128_1: "f32[1024]", arg129_1: "f32[1024, 512, 1, 1]", arg130_1: "f32[1024]", arg131_1: "f32[1024]", arg132_1: "f32[416, 1024, 1, 1]", arg133_1: "f32[416]", arg134_1: "f32[416]", arg135_1: "f32[104, 104, 3, 3]", arg136_1: "f32[104]", arg137_1: "f32[104]", arg138_1: "f32[104, 104, 3, 3]", arg139_1: "f32[104]", arg140_1: "f32[104]", arg141_1: "f32[104, 104, 3, 3]", arg142_1: "f32[104]", arg143_1: "f32[104]", arg144_1: "f32[1024, 416, 1, 1]", arg145_1: "f32[1024]", arg146_1: "f32[1024]", arg147_1: "f32[416, 1024, 1, 1]", arg148_1: "f32[416]", arg149_1: "f32[416]", arg150_1: "f32[104, 104, 3, 3]", arg151_1: "f32[104]", arg152_1: "f32[104]", arg153_1: "f32[104, 104, 3, 3]", arg154_1: "f32[104]", arg155_1: "f32[104]", arg156_1: "f32[104, 104, 3, 3]", arg157_1: "f32[104]", arg158_1: "f32[104]", arg159_1: "f32[1024, 416, 1, 1]", arg160_1: "f32[1024]", arg161_1: "f32[1024]", arg162_1: "f32[416, 1024, 1, 1]", arg163_1: "f32[416]", arg164_1: "f32[416]", arg165_1: "f32[104, 104, 3, 3]", arg166_1: "f32[104]", arg167_1: "f32[104]", arg168_1: "f32[104, 104, 3, 3]", arg169_1: "f32[104]", arg170_1: "f32[104]", arg171_1: "f32[104, 104, 3, 3]", arg172_1: "f32[104]", arg173_1: "f32[104]", arg174_1: "f32[1024, 416, 1, 1]", arg175_1: "f32[1024]", arg176_1: "f32[1024]", arg177_1: "f32[416, 1024, 1, 1]", arg178_1: "f32[416]", arg179_1: "f32[416]", arg180_1: "f32[104, 104, 3, 3]", arg181_1: "f32[104]", arg182_1: "f32[104]", arg183_1: "f32[104, 104, 3, 3]", arg184_1: "f32[104]", arg185_1: "f32[104]", arg186_1: "f32[104, 104, 3, 3]", arg187_1: "f32[104]", arg188_1: "f32[104]", arg189_1: "f32[1024, 416, 1, 1]", arg190_1: "f32[1024]", arg191_1: "f32[1024]", arg192_1: "f32[416, 1024, 1, 1]", arg193_1: "f32[416]", arg194_1: "f32[416]", arg195_1: "f32[104, 104, 3, 3]", arg196_1: "f32[104]", arg197_1: "f32[104]", arg198_1: "f32[104, 104, 3, 3]", arg199_1: "f32[104]", arg200_1: "f32[104]", arg201_1: "f32[104, 104, 3, 3]", arg202_1: "f32[104]", arg203_1: "f32[104]", arg204_1: "f32[1024, 416, 1, 1]", arg205_1: "f32[1024]", arg206_1: "f32[1024]", arg207_1: "f32[416, 1024, 1, 1]", arg208_1: "f32[416]", arg209_1: "f32[416]", arg210_1: "f32[104, 104, 3, 3]", arg211_1: "f32[104]", arg212_1: "f32[104]", arg213_1: "f32[104, 104, 3, 3]", arg214_1: "f32[104]", arg215_1: "f32[104]", arg216_1: "f32[104, 104, 3, 3]", arg217_1: "f32[104]", arg218_1: "f32[104]", arg219_1: "f32[1024, 416, 1, 1]", arg220_1: "f32[1024]", arg221_1: "f32[1024]", arg222_1: "f32[416, 1024, 1, 1]", arg223_1: "f32[416]", arg224_1: "f32[416]", arg225_1: "f32[104, 104, 3, 3]", arg226_1: "f32[104]", arg227_1: "f32[104]", arg228_1: "f32[104, 104, 3, 3]", arg229_1: "f32[104]", arg230_1: "f32[104]", arg231_1: "f32[104, 104, 3, 3]", arg232_1: "f32[104]", arg233_1: "f32[104]", arg234_1: "f32[1024, 416, 1, 1]", arg235_1: "f32[1024]", arg236_1: "f32[1024]", arg237_1: "f32[416, 1024, 1, 1]", arg238_1: "f32[416]", arg239_1: "f32[416]", arg240_1: "f32[104, 104, 3, 3]", arg241_1: "f32[104]", arg242_1: "f32[104]", arg243_1: "f32[104, 104, 3, 3]", arg244_1: "f32[104]", arg245_1: "f32[104]", arg246_1: "f32[104, 104, 3, 3]", arg247_1: "f32[104]", arg248_1: "f32[104]", arg249_1: "f32[1024, 416, 1, 1]", arg250_1: "f32[1024]", arg251_1: "f32[1024]", arg252_1: "f32[416, 1024, 1, 1]", arg253_1: "f32[416]", arg254_1: "f32[416]", arg255_1: "f32[104, 104, 3, 3]", arg256_1: "f32[104]", arg257_1: "f32[104]", arg258_1: "f32[104, 104, 3, 3]", arg259_1: "f32[104]", arg260_1: "f32[104]", arg261_1: "f32[104, 104, 3, 3]", arg262_1: "f32[104]", arg263_1: "f32[104]", arg264_1: "f32[1024, 416, 1, 1]", arg265_1: "f32[1024]", arg266_1: "f32[1024]", arg267_1: "f32[416, 1024, 1, 1]", arg268_1: "f32[416]", arg269_1: "f32[416]", arg270_1: "f32[104, 104, 3, 3]", arg271_1: "f32[104]", arg272_1: "f32[104]", arg273_1: "f32[104, 104, 3, 3]", arg274_1: "f32[104]", arg275_1: "f32[104]", arg276_1: "f32[104, 104, 3, 3]", arg277_1: "f32[104]", arg278_1: "f32[104]", arg279_1: "f32[1024, 416, 1, 1]", arg280_1: "f32[1024]", arg281_1: "f32[1024]", arg282_1: "f32[416, 1024, 1, 1]", arg283_1: "f32[416]", arg284_1: "f32[416]", arg285_1: "f32[104, 104, 3, 3]", arg286_1: "f32[104]", arg287_1: "f32[104]", arg288_1: "f32[104, 104, 3, 3]", arg289_1: "f32[104]", arg290_1: "f32[104]", arg291_1: "f32[104, 104, 3, 3]", arg292_1: "f32[104]", arg293_1: "f32[104]", arg294_1: "f32[1024, 416, 1, 1]", arg295_1: "f32[1024]", arg296_1: "f32[1024]", arg297_1: "f32[416, 1024, 1, 1]", arg298_1: "f32[416]", arg299_1: "f32[416]", arg300_1: "f32[104, 104, 3, 3]", arg301_1: "f32[104]", arg302_1: "f32[104]", arg303_1: "f32[104, 104, 3, 3]", arg304_1: "f32[104]", arg305_1: "f32[104]", arg306_1: "f32[104, 104, 3, 3]", arg307_1: "f32[104]", arg308_1: "f32[104]", arg309_1: "f32[1024, 416, 1, 1]", arg310_1: "f32[1024]", arg311_1: "f32[1024]", arg312_1: "f32[416, 1024, 1, 1]", arg313_1: "f32[416]", arg314_1: "f32[416]", arg315_1: "f32[104, 104, 3, 3]", arg316_1: "f32[104]", arg317_1: "f32[104]", arg318_1: "f32[104, 104, 3, 3]", arg319_1: "f32[104]", arg320_1: "f32[104]", arg321_1: "f32[104, 104, 3, 3]", arg322_1: "f32[104]", arg323_1: "f32[104]", arg324_1: "f32[1024, 416, 1, 1]", arg325_1: "f32[1024]", arg326_1: "f32[1024]", arg327_1: "f32[416, 1024, 1, 1]", arg328_1: "f32[416]", arg329_1: "f32[416]", arg330_1: "f32[104, 104, 3, 3]", arg331_1: "f32[104]", arg332_1: "f32[104]", arg333_1: "f32[104, 104, 3, 3]", arg334_1: "f32[104]", arg335_1: "f32[104]", arg336_1: "f32[104, 104, 3, 3]", arg337_1: "f32[104]", arg338_1: "f32[104]", arg339_1: "f32[1024, 416, 1, 1]", arg340_1: "f32[1024]", arg341_1: "f32[1024]", arg342_1: "f32[416, 1024, 1, 1]", arg343_1: "f32[416]", arg344_1: "f32[416]", arg345_1: "f32[104, 104, 3, 3]", arg346_1: "f32[104]", arg347_1: "f32[104]", arg348_1: "f32[104, 104, 3, 3]", arg349_1: "f32[104]", arg350_1: "f32[104]", arg351_1: "f32[104, 104, 3, 3]", arg352_1: "f32[104]", arg353_1: "f32[104]", arg354_1: "f32[1024, 416, 1, 1]", arg355_1: "f32[1024]", arg356_1: "f32[1024]", arg357_1: "f32[416, 1024, 1, 1]", arg358_1: "f32[416]", arg359_1: "f32[416]", arg360_1: "f32[104, 104, 3, 3]", arg361_1: "f32[104]", arg362_1: "f32[104]", arg363_1: "f32[104, 104, 3, 3]", arg364_1: "f32[104]", arg365_1: "f32[104]", arg366_1: "f32[104, 104, 3, 3]", arg367_1: "f32[104]", arg368_1: "f32[104]", arg369_1: "f32[1024, 416, 1, 1]", arg370_1: "f32[1024]", arg371_1: "f32[1024]", arg372_1: "f32[416, 1024, 1, 1]", arg373_1: "f32[416]", arg374_1: "f32[416]", arg375_1: "f32[104, 104, 3, 3]", arg376_1: "f32[104]", arg377_1: "f32[104]", arg378_1: "f32[104, 104, 3, 3]", arg379_1: "f32[104]", arg380_1: "f32[104]", arg381_1: "f32[104, 104, 3, 3]", arg382_1: "f32[104]", arg383_1: "f32[104]", arg384_1: "f32[1024, 416, 1, 1]", arg385_1: "f32[1024]", arg386_1: "f32[1024]", arg387_1: "f32[416, 1024, 1, 1]", arg388_1: "f32[416]", arg389_1: "f32[416]", arg390_1: "f32[104, 104, 3, 3]", arg391_1: "f32[104]", arg392_1: "f32[104]", arg393_1: "f32[104, 104, 3, 3]", arg394_1: "f32[104]", arg395_1: "f32[104]", arg396_1: "f32[104, 104, 3, 3]", arg397_1: "f32[104]", arg398_1: "f32[104]", arg399_1: "f32[1024, 416, 1, 1]", arg400_1: "f32[1024]", arg401_1: "f32[1024]", arg402_1: "f32[416, 1024, 1, 1]", arg403_1: "f32[416]", arg404_1: "f32[416]", arg405_1: "f32[104, 104, 3, 3]", arg406_1: "f32[104]", arg407_1: "f32[104]", arg408_1: "f32[104, 104, 3, 3]", arg409_1: "f32[104]", arg410_1: "f32[104]", arg411_1: "f32[104, 104, 3, 3]", arg412_1: "f32[104]", arg413_1: "f32[104]", arg414_1: "f32[1024, 416, 1, 1]", arg415_1: "f32[1024]", arg416_1: "f32[1024]", arg417_1: "f32[416, 1024, 1, 1]", arg418_1: "f32[416]", arg419_1: "f32[416]", arg420_1: "f32[104, 104, 3, 3]", arg421_1: "f32[104]", arg422_1: "f32[104]", arg423_1: "f32[104, 104, 3, 3]", arg424_1: "f32[104]", arg425_1: "f32[104]", arg426_1: "f32[104, 104, 3, 3]", arg427_1: "f32[104]", arg428_1: "f32[104]", arg429_1: "f32[1024, 416, 1, 1]", arg430_1: "f32[1024]", arg431_1: "f32[1024]", arg432_1: "f32[416, 1024, 1, 1]", arg433_1: "f32[416]", arg434_1: "f32[416]", arg435_1: "f32[104, 104, 3, 3]", arg436_1: "f32[104]", arg437_1: "f32[104]", arg438_1: "f32[104, 104, 3, 3]", arg439_1: "f32[104]", arg440_1: "f32[104]", arg441_1: "f32[104, 104, 3, 3]", arg442_1: "f32[104]", arg443_1: "f32[104]", arg444_1: "f32[1024, 416, 1, 1]", arg445_1: "f32[1024]", arg446_1: "f32[1024]", arg447_1: "f32[416, 1024, 1, 1]", arg448_1: "f32[416]", arg449_1: "f32[416]", arg450_1: "f32[104, 104, 3, 3]", arg451_1: "f32[104]", arg452_1: "f32[104]", arg453_1: "f32[104, 104, 3, 3]", arg454_1: "f32[104]", arg455_1: "f32[104]", arg456_1: "f32[104, 104, 3, 3]", arg457_1: "f32[104]", arg458_1: "f32[104]", arg459_1: "f32[1024, 416, 1, 1]", arg460_1: "f32[1024]", arg461_1: "f32[1024]", arg462_1: "f32[832, 1024, 1, 1]", arg463_1: "f32[832]", arg464_1: "f32[832]", arg465_1: "f32[208, 208, 3, 3]", arg466_1: "f32[208]", arg467_1: "f32[208]", arg468_1: "f32[208, 208, 3, 3]", arg469_1: "f32[208]", arg470_1: "f32[208]", arg471_1: "f32[208, 208, 3, 3]", arg472_1: "f32[208]", arg473_1: "f32[208]", arg474_1: "f32[2048, 832, 1, 1]", arg475_1: "f32[2048]", arg476_1: "f32[2048]", arg477_1: "f32[2048, 1024, 1, 1]", arg478_1: "f32[2048]", arg479_1: "f32[2048]", arg480_1: "f32[832, 2048, 1, 1]", arg481_1: "f32[832]", arg482_1: "f32[832]", arg483_1: "f32[208, 208, 3, 3]", arg484_1: "f32[208]", arg485_1: "f32[208]", arg486_1: "f32[208, 208, 3, 3]", arg487_1: "f32[208]", arg488_1: "f32[208]", arg489_1: "f32[208, 208, 3, 3]", arg490_1: "f32[208]", arg491_1: "f32[208]", arg492_1: "f32[2048, 832, 1, 1]", arg493_1: "f32[2048]", arg494_1: "f32[2048]", arg495_1: "f32[832, 2048, 1, 1]", arg496_1: "f32[832]", arg497_1: "f32[832]", arg498_1: "f32[208, 208, 3, 3]", arg499_1: "f32[208]", arg500_1: "f32[208]", arg501_1: "f32[208, 208, 3, 3]", arg502_1: "f32[208]", arg503_1: "f32[208]", arg504_1: "f32[208, 208, 3, 3]", arg505_1: "f32[208]", arg506_1: "f32[208]", arg507_1: "f32[2048, 832, 1, 1]", arg508_1: "f32[2048]", arg509_1: "f32[2048]", arg510_1: "f32[1000, 2048]", arg511_1: "f32[1000]", arg512_1: "f32[64]", arg513_1: "f32[64]", arg514_1: "i64[]", arg515_1: "f32[104]", arg516_1: "f32[104]", arg517_1: "i64[]", arg518_1: "f32[26]", arg519_1: "f32[26]", arg520_1: "i64[]", arg521_1: "f32[26]", arg522_1: "f32[26]", arg523_1: "i64[]", arg524_1: "f32[26]", arg525_1: "f32[26]", arg526_1: "i64[]", arg527_1: "f32[256]", arg528_1: "f32[256]", arg529_1: "i64[]", arg530_1: "f32[256]", arg531_1: "f32[256]", arg532_1: "i64[]", arg533_1: "f32[104]", arg534_1: "f32[104]", arg535_1: "i64[]", arg536_1: "f32[26]", arg537_1: "f32[26]", arg538_1: "i64[]", arg539_1: "f32[26]", arg540_1: "f32[26]", arg541_1: "i64[]", arg542_1: "f32[26]", arg543_1: "f32[26]", arg544_1: "i64[]", arg545_1: "f32[256]", arg546_1: "f32[256]", arg547_1: "i64[]", arg548_1: "f32[104]", arg549_1: "f32[104]", arg550_1: "i64[]", arg551_1: "f32[26]", arg552_1: "f32[26]", arg553_1: "i64[]", arg554_1: "f32[26]", arg555_1: "f32[26]", arg556_1: "i64[]", arg557_1: "f32[26]", arg558_1: "f32[26]", arg559_1: "i64[]", arg560_1: "f32[256]", arg561_1: "f32[256]", arg562_1: "i64[]", arg563_1: "f32[208]", arg564_1: "f32[208]", arg565_1: "i64[]", arg566_1: "f32[52]", arg567_1: "f32[52]", arg568_1: "i64[]", arg569_1: "f32[52]", arg570_1: "f32[52]", arg571_1: "i64[]", arg572_1: "f32[52]", arg573_1: "f32[52]", arg574_1: "i64[]", arg575_1: "f32[512]", arg576_1: "f32[512]", arg577_1: "i64[]", arg578_1: "f32[512]", arg579_1: "f32[512]", arg580_1: "i64[]", arg581_1: "f32[208]", arg582_1: "f32[208]", arg583_1: "i64[]", arg584_1: "f32[52]", arg585_1: "f32[52]", arg586_1: "i64[]", arg587_1: "f32[52]", arg588_1: "f32[52]", arg589_1: "i64[]", arg590_1: "f32[52]", arg591_1: "f32[52]", arg592_1: "i64[]", arg593_1: "f32[512]", arg594_1: "f32[512]", arg595_1: "i64[]", arg596_1: "f32[208]", arg597_1: "f32[208]", arg598_1: "i64[]", arg599_1: "f32[52]", arg600_1: "f32[52]", arg601_1: "i64[]", arg602_1: "f32[52]", arg603_1: "f32[52]", arg604_1: "i64[]", arg605_1: "f32[52]", arg606_1: "f32[52]", arg607_1: "i64[]", arg608_1: "f32[512]", arg609_1: "f32[512]", arg610_1: "i64[]", arg611_1: "f32[208]", arg612_1: "f32[208]", arg613_1: "i64[]", arg614_1: "f32[52]", arg615_1: "f32[52]", arg616_1: "i64[]", arg617_1: "f32[52]", arg618_1: "f32[52]", arg619_1: "i64[]", arg620_1: "f32[52]", arg621_1: "f32[52]", arg622_1: "i64[]", arg623_1: "f32[512]", arg624_1: "f32[512]", arg625_1: "i64[]", arg626_1: "f32[416]", arg627_1: "f32[416]", arg628_1: "i64[]", arg629_1: "f32[104]", arg630_1: "f32[104]", arg631_1: "i64[]", arg632_1: "f32[104]", arg633_1: "f32[104]", arg634_1: "i64[]", arg635_1: "f32[104]", arg636_1: "f32[104]", arg637_1: "i64[]", arg638_1: "f32[1024]", arg639_1: "f32[1024]", arg640_1: "i64[]", arg641_1: "f32[1024]", arg642_1: "f32[1024]", arg643_1: "i64[]", arg644_1: "f32[416]", arg645_1: "f32[416]", arg646_1: "i64[]", arg647_1: "f32[104]", arg648_1: "f32[104]", arg649_1: "i64[]", arg650_1: "f32[104]", arg651_1: "f32[104]", arg652_1: "i64[]", arg653_1: "f32[104]", arg654_1: "f32[104]", arg655_1: "i64[]", arg656_1: "f32[1024]", arg657_1: "f32[1024]", arg658_1: "i64[]", arg659_1: "f32[416]", arg660_1: "f32[416]", arg661_1: "i64[]", arg662_1: "f32[104]", arg663_1: "f32[104]", arg664_1: "i64[]", arg665_1: "f32[104]", arg666_1: "f32[104]", arg667_1: "i64[]", arg668_1: "f32[104]", arg669_1: "f32[104]", arg670_1: "i64[]", arg671_1: "f32[1024]", arg672_1: "f32[1024]", arg673_1: "i64[]", arg674_1: "f32[416]", arg675_1: "f32[416]", arg676_1: "i64[]", arg677_1: "f32[104]", arg678_1: "f32[104]", arg679_1: "i64[]", arg680_1: "f32[104]", arg681_1: "f32[104]", arg682_1: "i64[]", arg683_1: "f32[104]", arg684_1: "f32[104]", arg685_1: "i64[]", arg686_1: "f32[1024]", arg687_1: "f32[1024]", arg688_1: "i64[]", arg689_1: "f32[416]", arg690_1: "f32[416]", arg691_1: "i64[]", arg692_1: "f32[104]", arg693_1: "f32[104]", arg694_1: "i64[]", arg695_1: "f32[104]", arg696_1: "f32[104]", arg697_1: "i64[]", arg698_1: "f32[104]", arg699_1: "f32[104]", arg700_1: "i64[]", arg701_1: "f32[1024]", arg702_1: "f32[1024]", arg703_1: "i64[]", arg704_1: "f32[416]", arg705_1: "f32[416]", arg706_1: "i64[]", arg707_1: "f32[104]", arg708_1: "f32[104]", arg709_1: "i64[]", arg710_1: "f32[104]", arg711_1: "f32[104]", arg712_1: "i64[]", arg713_1: "f32[104]", arg714_1: "f32[104]", arg715_1: "i64[]", arg716_1: "f32[1024]", arg717_1: "f32[1024]", arg718_1: "i64[]", arg719_1: "f32[416]", arg720_1: "f32[416]", arg721_1: "i64[]", arg722_1: "f32[104]", arg723_1: "f32[104]", arg724_1: "i64[]", arg725_1: "f32[104]", arg726_1: "f32[104]", arg727_1: "i64[]", arg728_1: "f32[104]", arg729_1: "f32[104]", arg730_1: "i64[]", arg731_1: "f32[1024]", arg732_1: "f32[1024]", arg733_1: "i64[]", arg734_1: "f32[416]", arg735_1: "f32[416]", arg736_1: "i64[]", arg737_1: "f32[104]", arg738_1: "f32[104]", arg739_1: "i64[]", arg740_1: "f32[104]", arg741_1: "f32[104]", arg742_1: "i64[]", arg743_1: "f32[104]", arg744_1: "f32[104]", arg745_1: "i64[]", arg746_1: "f32[1024]", arg747_1: "f32[1024]", arg748_1: "i64[]", arg749_1: "f32[416]", arg750_1: "f32[416]", arg751_1: "i64[]", arg752_1: "f32[104]", arg753_1: "f32[104]", arg754_1: "i64[]", arg755_1: "f32[104]", arg756_1: "f32[104]", arg757_1: "i64[]", arg758_1: "f32[104]", arg759_1: "f32[104]", arg760_1: "i64[]", arg761_1: "f32[1024]", arg762_1: "f32[1024]", arg763_1: "i64[]", arg764_1: "f32[416]", arg765_1: "f32[416]", arg766_1: "i64[]", arg767_1: "f32[104]", arg768_1: "f32[104]", arg769_1: "i64[]", arg770_1: "f32[104]", arg771_1: "f32[104]", arg772_1: "i64[]", arg773_1: "f32[104]", arg774_1: "f32[104]", arg775_1: "i64[]", arg776_1: "f32[1024]", arg777_1: "f32[1024]", arg778_1: "i64[]", arg779_1: "f32[416]", arg780_1: "f32[416]", arg781_1: "i64[]", arg782_1: "f32[104]", arg783_1: "f32[104]", arg784_1: "i64[]", arg785_1: "f32[104]", arg786_1: "f32[104]", arg787_1: "i64[]", arg788_1: "f32[104]", arg789_1: "f32[104]", arg790_1: "i64[]", arg791_1: "f32[1024]", arg792_1: "f32[1024]", arg793_1: "i64[]", arg794_1: "f32[416]", arg795_1: "f32[416]", arg796_1: "i64[]", arg797_1: "f32[104]", arg798_1: "f32[104]", arg799_1: "i64[]", arg800_1: "f32[104]", arg801_1: "f32[104]", arg802_1: "i64[]", arg803_1: "f32[104]", arg804_1: "f32[104]", arg805_1: "i64[]", arg806_1: "f32[1024]", arg807_1: "f32[1024]", arg808_1: "i64[]", arg809_1: "f32[416]", arg810_1: "f32[416]", arg811_1: "i64[]", arg812_1: "f32[104]", arg813_1: "f32[104]", arg814_1: "i64[]", arg815_1: "f32[104]", arg816_1: "f32[104]", arg817_1: "i64[]", arg818_1: "f32[104]", arg819_1: "f32[104]", arg820_1: "i64[]", arg821_1: "f32[1024]", arg822_1: "f32[1024]", arg823_1: "i64[]", arg824_1: "f32[416]", arg825_1: "f32[416]", arg826_1: "i64[]", arg827_1: "f32[104]", arg828_1: "f32[104]", arg829_1: "i64[]", arg830_1: "f32[104]", arg831_1: "f32[104]", arg832_1: "i64[]", arg833_1: "f32[104]", arg834_1: "f32[104]", arg835_1: "i64[]", arg836_1: "f32[1024]", arg837_1: "f32[1024]", arg838_1: "i64[]", arg839_1: "f32[416]", arg840_1: "f32[416]", arg841_1: "i64[]", arg842_1: "f32[104]", arg843_1: "f32[104]", arg844_1: "i64[]", arg845_1: "f32[104]", arg846_1: "f32[104]", arg847_1: "i64[]", arg848_1: "f32[104]", arg849_1: "f32[104]", arg850_1: "i64[]", arg851_1: "f32[1024]", arg852_1: "f32[1024]", arg853_1: "i64[]", arg854_1: "f32[416]", arg855_1: "f32[416]", arg856_1: "i64[]", arg857_1: "f32[104]", arg858_1: "f32[104]", arg859_1: "i64[]", arg860_1: "f32[104]", arg861_1: "f32[104]", arg862_1: "i64[]", arg863_1: "f32[104]", arg864_1: "f32[104]", arg865_1: "i64[]", arg866_1: "f32[1024]", arg867_1: "f32[1024]", arg868_1: "i64[]", arg869_1: "f32[416]", arg870_1: "f32[416]", arg871_1: "i64[]", arg872_1: "f32[104]", arg873_1: "f32[104]", arg874_1: "i64[]", arg875_1: "f32[104]", arg876_1: "f32[104]", arg877_1: "i64[]", arg878_1: "f32[104]", arg879_1: "f32[104]", arg880_1: "i64[]", arg881_1: "f32[1024]", arg882_1: "f32[1024]", arg883_1: "i64[]", arg884_1: "f32[416]", arg885_1: "f32[416]", arg886_1: "i64[]", arg887_1: "f32[104]", arg888_1: "f32[104]", arg889_1: "i64[]", arg890_1: "f32[104]", arg891_1: "f32[104]", arg892_1: "i64[]", arg893_1: "f32[104]", arg894_1: "f32[104]", arg895_1: "i64[]", arg896_1: "f32[1024]", arg897_1: "f32[1024]", arg898_1: "i64[]", arg899_1: "f32[416]", arg900_1: "f32[416]", arg901_1: "i64[]", arg902_1: "f32[104]", arg903_1: "f32[104]", arg904_1: "i64[]", arg905_1: "f32[104]", arg906_1: "f32[104]", arg907_1: "i64[]", arg908_1: "f32[104]", arg909_1: "f32[104]", arg910_1: "i64[]", arg911_1: "f32[1024]", arg912_1: "f32[1024]", arg913_1: "i64[]", arg914_1: "f32[416]", arg915_1: "f32[416]", arg916_1: "i64[]", arg917_1: "f32[104]", arg918_1: "f32[104]", arg919_1: "i64[]", arg920_1: "f32[104]", arg921_1: "f32[104]", arg922_1: "i64[]", arg923_1: "f32[104]", arg924_1: "f32[104]", arg925_1: "i64[]", arg926_1: "f32[1024]", arg927_1: "f32[1024]", arg928_1: "i64[]", arg929_1: "f32[416]", arg930_1: "f32[416]", arg931_1: "i64[]", arg932_1: "f32[104]", arg933_1: "f32[104]", arg934_1: "i64[]", arg935_1: "f32[104]", arg936_1: "f32[104]", arg937_1: "i64[]", arg938_1: "f32[104]", arg939_1: "f32[104]", arg940_1: "i64[]", arg941_1: "f32[1024]", arg942_1: "f32[1024]", arg943_1: "i64[]", arg944_1: "f32[416]", arg945_1: "f32[416]", arg946_1: "i64[]", arg947_1: "f32[104]", arg948_1: "f32[104]", arg949_1: "i64[]", arg950_1: "f32[104]", arg951_1: "f32[104]", arg952_1: "i64[]", arg953_1: "f32[104]", arg954_1: "f32[104]", arg955_1: "i64[]", arg956_1: "f32[1024]", arg957_1: "f32[1024]", arg958_1: "i64[]", arg959_1: "f32[416]", arg960_1: "f32[416]", arg961_1: "i64[]", arg962_1: "f32[104]", arg963_1: "f32[104]", arg964_1: "i64[]", arg965_1: "f32[104]", arg966_1: "f32[104]", arg967_1: "i64[]", arg968_1: "f32[104]", arg969_1: "f32[104]", arg970_1: "i64[]", arg971_1: "f32[1024]", arg972_1: "f32[1024]", arg973_1: "i64[]", arg974_1: "f32[832]", arg975_1: "f32[832]", arg976_1: "i64[]", arg977_1: "f32[208]", arg978_1: "f32[208]", arg979_1: "i64[]", arg980_1: "f32[208]", arg981_1: "f32[208]", arg982_1: "i64[]", arg983_1: "f32[208]", arg984_1: "f32[208]", arg985_1: "i64[]", arg986_1: "f32[2048]", arg987_1: "f32[2048]", arg988_1: "i64[]", arg989_1: "f32[2048]", arg990_1: "f32[2048]", arg991_1: "i64[]", arg992_1: "f32[832]", arg993_1: "f32[832]", arg994_1: "i64[]", arg995_1: "f32[208]", arg996_1: "f32[208]", arg997_1: "i64[]", arg998_1: "f32[208]", arg999_1: "f32[208]", arg1000_1: "i64[]", arg1001_1: "f32[208]", arg1002_1: "f32[208]", arg1003_1: "i64[]", arg1004_1: "f32[2048]", arg1005_1: "f32[2048]", arg1006_1: "i64[]", arg1007_1: "f32[832]", arg1008_1: "f32[832]", arg1009_1: "i64[]", arg1010_1: "f32[208]", arg1011_1: "f32[208]", arg1012_1: "i64[]", arg1013_1: "f32[208]", arg1014_1: "f32[208]", arg1015_1: "i64[]", arg1016_1: "f32[208]", arg1017_1: "f32[208]", arg1018_1: "i64[]", arg1019_1: "f32[2048]", arg1020_1: "f32[2048]", arg1021_1: "i64[]", arg1022_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg1022_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1022_1 = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(arg512_1, torch.float32);  arg512_1 = None
    convert_element_type_1: "f32[64]" = torch.ops.prims.convert_element_type.default(arg513_1, torch.float32);  arg513_1 = None
    add: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    relu: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1]);  relu = None
    getitem: "f32[8, 64, 56, 56]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_1: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg3_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_2: "f32[104]" = torch.ops.prims.convert_element_type.default(arg515_1, torch.float32);  arg515_1 = None
    convert_element_type_3: "f32[104]" = torch.ops.prims.convert_element_type.default(arg516_1, torch.float32);  arg516_1 = None
    add_2: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[104]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_4: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_13: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_15: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_1: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(relu_1, [26, 26, 26, 26], 1)
    getitem_6: "f32[8, 26, 56, 56]" = split_with_sizes_1[0];  split_with_sizes_1 = None
    convolution_2: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_6, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_4: "f32[26]" = torch.ops.prims.convert_element_type.default(arg518_1, torch.float32);  arg518_1 = None
    convert_element_type_5: "f32[26]" = torch.ops.prims.convert_element_type.default(arg519_1, torch.float32);  arg519_1 = None
    add_4: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[26]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_7: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_21: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_23: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_2: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(relu_1, [26, 26, 26, 26], 1)
    getitem_11: "f32[8, 26, 56, 56]" = split_with_sizes_2[1];  split_with_sizes_2 = None
    convolution_3: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_11, arg9_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_11 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_6: "f32[26]" = torch.ops.prims.convert_element_type.default(arg521_1, torch.float32);  arg521_1 = None
    convert_element_type_7: "f32[26]" = torch.ops.prims.convert_element_type.default(arg522_1, torch.float32);  arg522_1 = None
    add_6: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[26]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    mul_10: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_29: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_31: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_3: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(relu_1, [26, 26, 26, 26], 1)
    getitem_16: "f32[8, 26, 56, 56]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    convolution_4: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_16, arg12_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_16 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_8: "f32[26]" = torch.ops.prims.convert_element_type.default(arg524_1, torch.float32);  arg524_1 = None
    convert_element_type_9: "f32[26]" = torch.ops.prims.convert_element_type.default(arg525_1, torch.float32);  arg525_1 = None
    add_8: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[26]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    mul_13: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_37: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_39: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_4: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(relu_1, [26, 26, 26, 26], 1);  relu_1 = None
    getitem_21: "f32[8, 26, 56, 56]" = split_with_sizes_4[3];  split_with_sizes_4 = None
    avg_pool2d: "f32[8, 26, 56, 56]" = torch.ops.aten.avg_pool2d.default(getitem_21, [3, 3], [1, 1], [1, 1]);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_2, relu_3, relu_4, avg_pool2d], 1);  relu_2 = relu_3 = relu_4 = avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_5: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_10: "f32[256]" = torch.ops.prims.convert_element_type.default(arg527_1, torch.float32);  arg527_1 = None
    convert_element_type_11: "f32[256]" = torch.ops.prims.convert_element_type.default(arg528_1, torch.float32);  arg528_1 = None
    add_10: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[256]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    mul_16: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_6: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg18_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg18_1 = None
    convert_element_type_12: "f32[256]" = torch.ops.prims.convert_element_type.default(arg530_1, torch.float32);  arg530_1 = None
    convert_element_type_13: "f32[256]" = torch.ops.prims.convert_element_type.default(arg531_1, torch.float32);  arg531_1 = None
    add_12: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[256]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    mul_19: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_14: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_11, add_13);  add_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_5: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_7: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(relu_5, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_14: "f32[104]" = torch.ops.prims.convert_element_type.default(arg533_1, torch.float32);  arg533_1 = None
    convert_element_type_15: "f32[104]" = torch.ops.prims.convert_element_type.default(arg534_1, torch.float32);  arg534_1 = None
    add_15: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[104]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    mul_22: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_61: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_63: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_6: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(relu_6, [26, 26, 26, 26], 1)
    getitem_26: "f32[8, 26, 56, 56]" = split_with_sizes_6[0];  split_with_sizes_6 = None
    convolution_8: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_26, arg24_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_26 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_16: "f32[26]" = torch.ops.prims.convert_element_type.default(arg536_1, torch.float32);  arg536_1 = None
    convert_element_type_17: "f32[26]" = torch.ops.prims.convert_element_type.default(arg537_1, torch.float32);  arg537_1 = None
    add_17: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[26]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_8: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    mul_25: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_69: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_71: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_18: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_7: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(relu_6, [26, 26, 26, 26], 1)
    getitem_31: "f32[8, 26, 56, 56]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    add_19: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_7, getitem_31);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_9: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_19, arg27_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_19 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_18: "f32[26]" = torch.ops.prims.convert_element_type.default(arg539_1, torch.float32);  arg539_1 = None
    convert_element_type_19: "f32[26]" = torch.ops.prims.convert_element_type.default(arg540_1, torch.float32);  arg540_1 = None
    add_20: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[26]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_9: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    mul_28: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_77: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_79: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_21: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_8: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(relu_6, [26, 26, 26, 26], 1)
    getitem_36: "f32[8, 26, 56, 56]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    add_22: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_8, getitem_36);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_10: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_22, arg30_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_22 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_20: "f32[26]" = torch.ops.prims.convert_element_type.default(arg542_1, torch.float32);  arg542_1 = None
    convert_element_type_21: "f32[26]" = torch.ops.prims.convert_element_type.default(arg543_1, torch.float32);  arg543_1 = None
    add_23: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[26]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_10: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    mul_31: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_85: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_87: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_24: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_9: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(relu_6, [26, 26, 26, 26], 1);  relu_6 = None
    getitem_41: "f32[8, 26, 56, 56]" = split_with_sizes_9[3];  split_with_sizes_9 = None
    cat_1: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_7, relu_8, relu_9, getitem_41], 1);  relu_7 = relu_8 = relu_9 = getitem_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_11: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_1, arg33_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_1 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_22: "f32[256]" = torch.ops.prims.convert_element_type.default(arg545_1, torch.float32);  arg545_1 = None
    convert_element_type_23: "f32[256]" = torch.ops.prims.convert_element_type.default(arg546_1, torch.float32);  arg546_1 = None
    add_25: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[256]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_11: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    mul_34: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_26: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_27: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_26, relu_5);  add_26 = relu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_10: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_12: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(relu_10, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_24: "f32[104]" = torch.ops.prims.convert_element_type.default(arg548_1, torch.float32);  arg548_1 = None
    convert_element_type_25: "f32[104]" = torch.ops.prims.convert_element_type.default(arg549_1, torch.float32);  arg549_1 = None
    add_28: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[104]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_12: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    mul_37: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_101: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_103: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_29: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_11: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(relu_11, [26, 26, 26, 26], 1)
    getitem_46: "f32[8, 26, 56, 56]" = split_with_sizes_11[0];  split_with_sizes_11 = None
    convolution_13: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_46, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_46 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_26: "f32[26]" = torch.ops.prims.convert_element_type.default(arg551_1, torch.float32);  arg551_1 = None
    convert_element_type_27: "f32[26]" = torch.ops.prims.convert_element_type.default(arg552_1, torch.float32);  arg552_1 = None
    add_30: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[26]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_13: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    mul_40: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_109: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_111: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_31: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_12: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(relu_11, [26, 26, 26, 26], 1)
    getitem_51: "f32[8, 26, 56, 56]" = split_with_sizes_12[1];  split_with_sizes_12 = None
    add_32: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_12, getitem_51);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_14: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_32, arg42_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_32 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_28: "f32[26]" = torch.ops.prims.convert_element_type.default(arg554_1, torch.float32);  arg554_1 = None
    convert_element_type_29: "f32[26]" = torch.ops.prims.convert_element_type.default(arg555_1, torch.float32);  arg555_1 = None
    add_33: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[26]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_14: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    mul_43: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_117: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_119: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_34: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_13: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(relu_11, [26, 26, 26, 26], 1)
    getitem_56: "f32[8, 26, 56, 56]" = split_with_sizes_13[2];  split_with_sizes_13 = None
    add_35: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_13, getitem_56);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_15: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_35, arg45_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_35 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_30: "f32[26]" = torch.ops.prims.convert_element_type.default(arg557_1, torch.float32);  arg557_1 = None
    convert_element_type_31: "f32[26]" = torch.ops.prims.convert_element_type.default(arg558_1, torch.float32);  arg558_1 = None
    add_36: "f32[26]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[26]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_15: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    mul_46: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_125: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_127: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_37: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_14: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(relu_11, [26, 26, 26, 26], 1);  relu_11 = None
    getitem_61: "f32[8, 26, 56, 56]" = split_with_sizes_14[3];  split_with_sizes_14 = None
    cat_2: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_12, relu_13, relu_14, getitem_61], 1);  relu_12 = relu_13 = relu_14 = getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_16: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_2, arg48_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_32: "f32[256]" = torch.ops.prims.convert_element_type.default(arg560_1, torch.float32);  arg560_1 = None
    convert_element_type_33: "f32[256]" = torch.ops.prims.convert_element_type.default(arg561_1, torch.float32);  arg561_1 = None
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[256]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_16: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    mul_49: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_133: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_135: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_39: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_40: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_39, relu_10);  add_39 = relu_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_15: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_17: "f32[8, 208, 56, 56]" = torch.ops.aten.convolution.default(relu_15, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_34: "f32[208]" = torch.ops.prims.convert_element_type.default(arg563_1, torch.float32);  arg563_1 = None
    convert_element_type_35: "f32[208]" = torch.ops.prims.convert_element_type.default(arg564_1, torch.float32);  arg564_1 = None
    add_41: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[208]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_17: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[8, 208, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    mul_52: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_141: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_143: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_42: "f32[8, 208, 56, 56]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_16: "f32[8, 208, 56, 56]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(relu_16, [52, 52, 52, 52], 1)
    getitem_66: "f32[8, 52, 56, 56]" = split_with_sizes_16[0];  split_with_sizes_16 = None
    convolution_18: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_66, arg54_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_66 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_36: "f32[52]" = torch.ops.prims.convert_element_type.default(arg566_1, torch.float32);  arg566_1 = None
    convert_element_type_37: "f32[52]" = torch.ops.prims.convert_element_type.default(arg567_1, torch.float32);  arg567_1 = None
    add_43: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[52]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_18: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    mul_55: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_149: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_151: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_44: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_17: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(relu_16, [52, 52, 52, 52], 1)
    getitem_71: "f32[8, 52, 56, 56]" = split_with_sizes_17[1];  split_with_sizes_17 = None
    convolution_19: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_71, arg57_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_71 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_38: "f32[52]" = torch.ops.prims.convert_element_type.default(arg569_1, torch.float32);  arg569_1 = None
    convert_element_type_39: "f32[52]" = torch.ops.prims.convert_element_type.default(arg570_1, torch.float32);  arg570_1 = None
    add_45: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[52]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_19: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
    mul_58: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_157: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_159: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_46: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_18: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_18 = torch.ops.aten.split_with_sizes.default(relu_16, [52, 52, 52, 52], 1)
    getitem_76: "f32[8, 52, 56, 56]" = split_with_sizes_18[2];  split_with_sizes_18 = None
    convolution_20: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_76, arg60_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_76 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_40: "f32[52]" = torch.ops.prims.convert_element_type.default(arg572_1, torch.float32);  arg572_1 = None
    convert_element_type_41: "f32[52]" = torch.ops.prims.convert_element_type.default(arg573_1, torch.float32);  arg573_1 = None
    add_47: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[52]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_20: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
    mul_61: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_165: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_167: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_48: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_19: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_48);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(relu_16, [52, 52, 52, 52], 1);  relu_16 = None
    getitem_81: "f32[8, 52, 56, 56]" = split_with_sizes_19[3];  split_with_sizes_19 = None
    avg_pool2d_1: "f32[8, 52, 28, 28]" = torch.ops.aten.avg_pool2d.default(getitem_81, [3, 3], [2, 2], [1, 1]);  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_3: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_17, relu_18, relu_19, avg_pool2d_1], 1);  relu_17 = relu_18 = relu_19 = avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_21: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_3, arg63_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_3 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_42: "f32[512]" = torch.ops.prims.convert_element_type.default(arg575_1, torch.float32);  arg575_1 = None
    convert_element_type_43: "f32[512]" = torch.ops.prims.convert_element_type.default(arg576_1, torch.float32);  arg576_1 = None
    add_49: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[512]" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
    reciprocal_21: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
    mul_64: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_173: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_175: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_50: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_22: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_15, arg66_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg66_1 = None
    convert_element_type_44: "f32[512]" = torch.ops.prims.convert_element_type.default(arg578_1, torch.float32);  arg578_1 = None
    convert_element_type_45: "f32[512]" = torch.ops.prims.convert_element_type.default(arg579_1, torch.float32);  arg579_1 = None
    add_51: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[512]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    reciprocal_22: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
    mul_67: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_181: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_183: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_52: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_53: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_50, add_52);  add_50 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_20: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_53);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_23: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_20, arg69_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_46: "f32[208]" = torch.ops.prims.convert_element_type.default(arg581_1, torch.float32);  arg581_1 = None
    convert_element_type_47: "f32[208]" = torch.ops.prims.convert_element_type.default(arg582_1, torch.float32);  arg582_1 = None
    add_54: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[208]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_23: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
    mul_70: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_189: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_191: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_55: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_21: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_55);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(relu_21, [52, 52, 52, 52], 1)
    getitem_86: "f32[8, 52, 28, 28]" = split_with_sizes_21[0];  split_with_sizes_21 = None
    convolution_24: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_86, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_86 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_48: "f32[52]" = torch.ops.prims.convert_element_type.default(arg584_1, torch.float32);  arg584_1 = None
    convert_element_type_49: "f32[52]" = torch.ops.prims.convert_element_type.default(arg585_1, torch.float32);  arg585_1 = None
    add_56: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[52]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_24: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
    mul_73: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_197: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_199: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_57: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_22: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(relu_21, [52, 52, 52, 52], 1)
    getitem_91: "f32[8, 52, 28, 28]" = split_with_sizes_22[1];  split_with_sizes_22 = None
    add_58: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_22, getitem_91);  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_25: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_58, arg75_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_58 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_50: "f32[52]" = torch.ops.prims.convert_element_type.default(arg587_1, torch.float32);  arg587_1 = None
    convert_element_type_51: "f32[52]" = torch.ops.prims.convert_element_type.default(arg588_1, torch.float32);  arg588_1 = None
    add_59: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[52]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_25: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
    mul_76: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_205: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_207: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_60: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_23: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_60);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_23 = torch.ops.aten.split_with_sizes.default(relu_21, [52, 52, 52, 52], 1)
    getitem_96: "f32[8, 52, 28, 28]" = split_with_sizes_23[2];  split_with_sizes_23 = None
    add_61: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_23, getitem_96);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_26: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_61, arg78_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_61 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_52: "f32[52]" = torch.ops.prims.convert_element_type.default(arg590_1, torch.float32);  arg590_1 = None
    convert_element_type_53: "f32[52]" = torch.ops.prims.convert_element_type.default(arg591_1, torch.float32);  arg591_1 = None
    add_62: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[52]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_26: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
    mul_79: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_213: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_215: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_63: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_24: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(relu_21, [52, 52, 52, 52], 1);  relu_21 = None
    getitem_101: "f32[8, 52, 28, 28]" = split_with_sizes_24[3];  split_with_sizes_24 = None
    cat_4: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_22, relu_23, relu_24, getitem_101], 1);  relu_22 = relu_23 = relu_24 = getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_27: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_4, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_54: "f32[512]" = torch.ops.prims.convert_element_type.default(arg593_1, torch.float32);  arg593_1 = None
    convert_element_type_55: "f32[512]" = torch.ops.prims.convert_element_type.default(arg594_1, torch.float32);  arg594_1 = None
    add_64: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[512]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_27: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  convolution_27 = unsqueeze_217 = None
    mul_82: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_221: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_223: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_65: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_66: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_65, relu_20);  add_65 = relu_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_25: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_66);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_28: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_25, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_56: "f32[208]" = torch.ops.prims.convert_element_type.default(arg596_1, torch.float32);  arg596_1 = None
    convert_element_type_57: "f32[208]" = torch.ops.prims.convert_element_type.default(arg597_1, torch.float32);  arg597_1 = None
    add_67: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[208]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_28: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  convolution_28 = unsqueeze_225 = None
    mul_85: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_229: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_231: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_68: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_26: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_68);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(relu_26, [52, 52, 52, 52], 1)
    getitem_106: "f32[8, 52, 28, 28]" = split_with_sizes_26[0];  split_with_sizes_26 = None
    convolution_29: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_106, arg87_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_106 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_58: "f32[52]" = torch.ops.prims.convert_element_type.default(arg599_1, torch.float32);  arg599_1 = None
    convert_element_type_59: "f32[52]" = torch.ops.prims.convert_element_type.default(arg600_1, torch.float32);  arg600_1 = None
    add_69: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[52]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_29: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  convolution_29 = unsqueeze_233 = None
    mul_88: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_237: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_239: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_70: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_27: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_70);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(relu_26, [52, 52, 52, 52], 1)
    getitem_111: "f32[8, 52, 28, 28]" = split_with_sizes_27[1];  split_with_sizes_27 = None
    add_71: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_27, getitem_111);  getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_30: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_71, arg90_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_71 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_60: "f32[52]" = torch.ops.prims.convert_element_type.default(arg602_1, torch.float32);  arg602_1 = None
    convert_element_type_61: "f32[52]" = torch.ops.prims.convert_element_type.default(arg603_1, torch.float32);  arg603_1 = None
    add_72: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[52]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_30: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  convolution_30 = unsqueeze_241 = None
    mul_91: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_245: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_247: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_73: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_28: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_28 = torch.ops.aten.split_with_sizes.default(relu_26, [52, 52, 52, 52], 1)
    getitem_116: "f32[8, 52, 28, 28]" = split_with_sizes_28[2];  split_with_sizes_28 = None
    add_74: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_28, getitem_116);  getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_31: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_74, arg93_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_74 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_62: "f32[52]" = torch.ops.prims.convert_element_type.default(arg605_1, torch.float32);  arg605_1 = None
    convert_element_type_63: "f32[52]" = torch.ops.prims.convert_element_type.default(arg606_1, torch.float32);  arg606_1 = None
    add_75: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[52]" = torch.ops.aten.sqrt.default(add_75);  add_75 = None
    reciprocal_31: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
    mul_94: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_253: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_255: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_76: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_29: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_76);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_29 = torch.ops.aten.split_with_sizes.default(relu_26, [52, 52, 52, 52], 1);  relu_26 = None
    getitem_121: "f32[8, 52, 28, 28]" = split_with_sizes_29[3];  split_with_sizes_29 = None
    cat_5: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_27, relu_28, relu_29, getitem_121], 1);  relu_27 = relu_28 = relu_29 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_32: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_5, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_64: "f32[512]" = torch.ops.prims.convert_element_type.default(arg608_1, torch.float32);  arg608_1 = None
    convert_element_type_65: "f32[512]" = torch.ops.prims.convert_element_type.default(arg609_1, torch.float32);  arg609_1 = None
    add_77: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[512]" = torch.ops.aten.sqrt.default(add_77);  add_77 = None
    reciprocal_32: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
    mul_97: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_261: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_263: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_78: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_79: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_78, relu_25);  add_78 = relu_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_30: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_33: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_30, arg99_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_66: "f32[208]" = torch.ops.prims.convert_element_type.default(arg611_1, torch.float32);  arg611_1 = None
    convert_element_type_67: "f32[208]" = torch.ops.prims.convert_element_type.default(arg612_1, torch.float32);  arg612_1 = None
    add_80: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[208]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_33: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
    mul_100: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_269: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_271: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_81: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_31: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_31 = torch.ops.aten.split_with_sizes.default(relu_31, [52, 52, 52, 52], 1)
    getitem_126: "f32[8, 52, 28, 28]" = split_with_sizes_31[0];  split_with_sizes_31 = None
    convolution_34: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_126, arg102_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_126 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_68: "f32[52]" = torch.ops.prims.convert_element_type.default(arg614_1, torch.float32);  arg614_1 = None
    convert_element_type_69: "f32[52]" = torch.ops.prims.convert_element_type.default(arg615_1, torch.float32);  arg615_1 = None
    add_82: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[52]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_34: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
    mul_103: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_277: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_279: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_83: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_32: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_83);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_32 = torch.ops.aten.split_with_sizes.default(relu_31, [52, 52, 52, 52], 1)
    getitem_131: "f32[8, 52, 28, 28]" = split_with_sizes_32[1];  split_with_sizes_32 = None
    add_84: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_32, getitem_131);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_35: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_84, arg105_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_84 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_70: "f32[52]" = torch.ops.prims.convert_element_type.default(arg617_1, torch.float32);  arg617_1 = None
    convert_element_type_71: "f32[52]" = torch.ops.prims.convert_element_type.default(arg618_1, torch.float32);  arg618_1 = None
    add_85: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[52]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_35: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
    mul_106: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_285: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_287: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_86: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_33: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_33 = torch.ops.aten.split_with_sizes.default(relu_31, [52, 52, 52, 52], 1)
    getitem_136: "f32[8, 52, 28, 28]" = split_with_sizes_33[2];  split_with_sizes_33 = None
    add_87: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_33, getitem_136);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_36: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_87, arg108_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_87 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_72: "f32[52]" = torch.ops.prims.convert_element_type.default(arg620_1, torch.float32);  arg620_1 = None
    convert_element_type_73: "f32[52]" = torch.ops.prims.convert_element_type.default(arg621_1, torch.float32);  arg621_1 = None
    add_88: "f32[52]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[52]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_36: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
    mul_109: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_293: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_295: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_89: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_34: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_89);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_34 = torch.ops.aten.split_with_sizes.default(relu_31, [52, 52, 52, 52], 1);  relu_31 = None
    getitem_141: "f32[8, 52, 28, 28]" = split_with_sizes_34[3];  split_with_sizes_34 = None
    cat_6: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_32, relu_33, relu_34, getitem_141], 1);  relu_32 = relu_33 = relu_34 = getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_37: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_6, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_74: "f32[512]" = torch.ops.prims.convert_element_type.default(arg623_1, torch.float32);  arg623_1 = None
    convert_element_type_75: "f32[512]" = torch.ops.prims.convert_element_type.default(arg624_1, torch.float32);  arg624_1 = None
    add_90: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[512]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_37: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
    mul_112: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_301: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_303: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_91: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_92: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_91, relu_30);  add_91 = relu_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_35: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_92);  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_38: "f32[8, 416, 28, 28]" = torch.ops.aten.convolution.default(relu_35, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_76: "f32[416]" = torch.ops.prims.convert_element_type.default(arg626_1, torch.float32);  arg626_1 = None
    convert_element_type_77: "f32[416]" = torch.ops.prims.convert_element_type.default(arg627_1, torch.float32);  arg627_1 = None
    add_93: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[416]" = torch.ops.aten.sqrt.default(add_93);  add_93 = None
    reciprocal_38: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[8, 416, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
    mul_115: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_309: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_311: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_94: "f32[8, 416, 28, 28]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_36: "f32[8, 416, 28, 28]" = torch.ops.aten.relu.default(add_94);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_36 = torch.ops.aten.split_with_sizes.default(relu_36, [104, 104, 104, 104], 1)
    getitem_146: "f32[8, 104, 28, 28]" = split_with_sizes_36[0];  split_with_sizes_36 = None
    convolution_39: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_146, arg117_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_146 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_78: "f32[104]" = torch.ops.prims.convert_element_type.default(arg629_1, torch.float32);  arg629_1 = None
    convert_element_type_79: "f32[104]" = torch.ops.prims.convert_element_type.default(arg630_1, torch.float32);  arg630_1 = None
    add_95: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[104]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
    reciprocal_39: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
    mul_118: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_317: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_319: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_96: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_37: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_96);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_37 = torch.ops.aten.split_with_sizes.default(relu_36, [104, 104, 104, 104], 1)
    getitem_151: "f32[8, 104, 28, 28]" = split_with_sizes_37[1];  split_with_sizes_37 = None
    convolution_40: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_151, arg120_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_151 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_80: "f32[104]" = torch.ops.prims.convert_element_type.default(arg632_1, torch.float32);  arg632_1 = None
    convert_element_type_81: "f32[104]" = torch.ops.prims.convert_element_type.default(arg633_1, torch.float32);  arg633_1 = None
    add_97: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[104]" = torch.ops.aten.sqrt.default(add_97);  add_97 = None
    reciprocal_40: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
    mul_121: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_325: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_327: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_98: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_38: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_98);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_38 = torch.ops.aten.split_with_sizes.default(relu_36, [104, 104, 104, 104], 1)
    getitem_156: "f32[8, 104, 28, 28]" = split_with_sizes_38[2];  split_with_sizes_38 = None
    convolution_41: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_156, arg123_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_156 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_82: "f32[104]" = torch.ops.prims.convert_element_type.default(arg635_1, torch.float32);  arg635_1 = None
    convert_element_type_83: "f32[104]" = torch.ops.prims.convert_element_type.default(arg636_1, torch.float32);  arg636_1 = None
    add_99: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[104]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_41: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
    mul_124: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_333: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_335: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_100: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_39: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_39 = torch.ops.aten.split_with_sizes.default(relu_36, [104, 104, 104, 104], 1);  relu_36 = None
    getitem_161: "f32[8, 104, 28, 28]" = split_with_sizes_39[3];  split_with_sizes_39 = None
    avg_pool2d_2: "f32[8, 104, 14, 14]" = torch.ops.aten.avg_pool2d.default(getitem_161, [3, 3], [2, 2], [1, 1]);  getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_7: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_37, relu_38, relu_39, avg_pool2d_2], 1);  relu_37 = relu_38 = relu_39 = avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_7, arg126_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_84: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg638_1, torch.float32);  arg638_1 = None
    convert_element_type_85: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg639_1, torch.float32);  arg639_1 = None
    add_101: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[1024]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_42: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
    mul_127: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_341: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_343: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_102: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_43: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, arg129_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = arg129_1 = None
    convert_element_type_86: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg641_1, torch.float32);  arg641_1 = None
    convert_element_type_87: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg642_1, torch.float32);  arg642_1 = None
    add_103: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[1024]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_43: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
    mul_130: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_349: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_351: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_104: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_105: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_102, add_104);  add_102 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_40: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_44: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_40, arg132_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_88: "f32[416]" = torch.ops.prims.convert_element_type.default(arg644_1, torch.float32);  arg644_1 = None
    convert_element_type_89: "f32[416]" = torch.ops.prims.convert_element_type.default(arg645_1, torch.float32);  arg645_1 = None
    add_106: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[416]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_44: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
    mul_133: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_357: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
    unsqueeze_359: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_107: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_41: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_41 = torch.ops.aten.split_with_sizes.default(relu_41, [104, 104, 104, 104], 1)
    getitem_166: "f32[8, 104, 14, 14]" = split_with_sizes_41[0];  split_with_sizes_41 = None
    convolution_45: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_166, arg135_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_166 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_90: "f32[104]" = torch.ops.prims.convert_element_type.default(arg647_1, torch.float32);  arg647_1 = None
    convert_element_type_91: "f32[104]" = torch.ops.prims.convert_element_type.default(arg648_1, torch.float32);  arg648_1 = None
    add_108: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[104]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_45: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
    mul_136: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
    unsqueeze_365: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_367: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_109: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_42: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_42 = torch.ops.aten.split_with_sizes.default(relu_41, [104, 104, 104, 104], 1)
    getitem_171: "f32[8, 104, 14, 14]" = split_with_sizes_42[1];  split_with_sizes_42 = None
    add_110: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_42, getitem_171);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_46: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_110, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_110 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_92: "f32[104]" = torch.ops.prims.convert_element_type.default(arg650_1, torch.float32);  arg650_1 = None
    convert_element_type_93: "f32[104]" = torch.ops.prims.convert_element_type.default(arg651_1, torch.float32);  arg651_1 = None
    add_111: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[104]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
    reciprocal_46: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
    mul_139: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_373: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
    unsqueeze_375: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_112: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_43: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_112);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_43 = torch.ops.aten.split_with_sizes.default(relu_41, [104, 104, 104, 104], 1)
    getitem_176: "f32[8, 104, 14, 14]" = split_with_sizes_43[2];  split_with_sizes_43 = None
    add_113: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_43, getitem_176);  getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_47: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_113, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_113 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_94: "f32[104]" = torch.ops.prims.convert_element_type.default(arg653_1, torch.float32);  arg653_1 = None
    convert_element_type_95: "f32[104]" = torch.ops.prims.convert_element_type.default(arg654_1, torch.float32);  arg654_1 = None
    add_114: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[104]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
    reciprocal_47: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
    mul_142: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_381: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_383: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_115: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_44: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_115);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_44 = torch.ops.aten.split_with_sizes.default(relu_41, [104, 104, 104, 104], 1);  relu_41 = None
    getitem_181: "f32[8, 104, 14, 14]" = split_with_sizes_44[3];  split_with_sizes_44 = None
    cat_8: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_42, relu_43, relu_44, getitem_181], 1);  relu_42 = relu_43 = relu_44 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_8, arg144_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_8 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_96: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg656_1, torch.float32);  arg656_1 = None
    convert_element_type_97: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg657_1, torch.float32);  arg657_1 = None
    add_116: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[1024]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
    reciprocal_48: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
    mul_145: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
    unsqueeze_389: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
    unsqueeze_391: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_117: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_118: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_117, relu_40);  add_117 = relu_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_118);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_49: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_45, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_98: "f32[416]" = torch.ops.prims.convert_element_type.default(arg659_1, torch.float32);  arg659_1 = None
    convert_element_type_99: "f32[416]" = torch.ops.prims.convert_element_type.default(arg660_1, torch.float32);  arg660_1 = None
    add_119: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[416]" = torch.ops.aten.sqrt.default(add_119);  add_119 = None
    reciprocal_49: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
    mul_148: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
    unsqueeze_397: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
    unsqueeze_399: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_120: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_46: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_120);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_46 = torch.ops.aten.split_with_sizes.default(relu_46, [104, 104, 104, 104], 1)
    getitem_186: "f32[8, 104, 14, 14]" = split_with_sizes_46[0];  split_with_sizes_46 = None
    convolution_50: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_186, arg150_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_186 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_100: "f32[104]" = torch.ops.prims.convert_element_type.default(arg662_1, torch.float32);  arg662_1 = None
    convert_element_type_101: "f32[104]" = torch.ops.prims.convert_element_type.default(arg663_1, torch.float32);  arg663_1 = None
    add_121: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[104]" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
    reciprocal_50: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
    mul_151: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
    unsqueeze_405: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
    unsqueeze_407: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_122: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_47: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_122);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_47 = torch.ops.aten.split_with_sizes.default(relu_46, [104, 104, 104, 104], 1)
    getitem_191: "f32[8, 104, 14, 14]" = split_with_sizes_47[1];  split_with_sizes_47 = None
    add_123: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_47, getitem_191);  getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_51: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_123, arg153_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_123 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_102: "f32[104]" = torch.ops.prims.convert_element_type.default(arg665_1, torch.float32);  arg665_1 = None
    convert_element_type_103: "f32[104]" = torch.ops.prims.convert_element_type.default(arg666_1, torch.float32);  arg666_1 = None
    add_124: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[104]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_51: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
    mul_154: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
    unsqueeze_413: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
    unsqueeze_415: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_125: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_48: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_125);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_48 = torch.ops.aten.split_with_sizes.default(relu_46, [104, 104, 104, 104], 1)
    getitem_196: "f32[8, 104, 14, 14]" = split_with_sizes_48[2];  split_with_sizes_48 = None
    add_126: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_48, getitem_196);  getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_52: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_126, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_126 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_104: "f32[104]" = torch.ops.prims.convert_element_type.default(arg668_1, torch.float32);  arg668_1 = None
    convert_element_type_105: "f32[104]" = torch.ops.prims.convert_element_type.default(arg669_1, torch.float32);  arg669_1 = None
    add_127: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[104]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_52: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
    mul_157: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
    unsqueeze_421: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
    unsqueeze_423: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_128: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_49: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_49 = torch.ops.aten.split_with_sizes.default(relu_46, [104, 104, 104, 104], 1);  relu_46 = None
    getitem_201: "f32[8, 104, 14, 14]" = split_with_sizes_49[3];  split_with_sizes_49 = None
    cat_9: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_47, relu_48, relu_49, getitem_201], 1);  relu_47 = relu_48 = relu_49 = getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_53: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_9, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_9 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_106: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg671_1, torch.float32);  arg671_1 = None
    convert_element_type_107: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg672_1, torch.float32);  arg672_1 = None
    add_129: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[1024]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
    reciprocal_53: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_159: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_427: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
    mul_160: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
    unsqueeze_429: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_161: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
    unsqueeze_430: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
    unsqueeze_431: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_130: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_131: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_130, relu_45);  add_130 = relu_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_50: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_131);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_54: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_50, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_108: "f32[416]" = torch.ops.prims.convert_element_type.default(arg674_1, torch.float32);  arg674_1 = None
    convert_element_type_109: "f32[416]" = torch.ops.prims.convert_element_type.default(arg675_1, torch.float32);  arg675_1 = None
    add_132: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[416]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
    reciprocal_54: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_162: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_435: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
    mul_163: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
    unsqueeze_437: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_164: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
    unsqueeze_438: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
    unsqueeze_439: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_133: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_51: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_51 = torch.ops.aten.split_with_sizes.default(relu_51, [104, 104, 104, 104], 1)
    getitem_206: "f32[8, 104, 14, 14]" = split_with_sizes_51[0];  split_with_sizes_51 = None
    convolution_55: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_206, arg165_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_206 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_110: "f32[104]" = torch.ops.prims.convert_element_type.default(arg677_1, torch.float32);  arg677_1 = None
    convert_element_type_111: "f32[104]" = torch.ops.prims.convert_element_type.default(arg678_1, torch.float32);  arg678_1 = None
    add_134: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[104]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
    reciprocal_55: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_165: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_443: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
    mul_166: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
    unsqueeze_445: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_167: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
    unsqueeze_446: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
    unsqueeze_447: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_135: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_52: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_135);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_52 = torch.ops.aten.split_with_sizes.default(relu_51, [104, 104, 104, 104], 1)
    getitem_211: "f32[8, 104, 14, 14]" = split_with_sizes_52[1];  split_with_sizes_52 = None
    add_136: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_52, getitem_211);  getitem_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_56: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_136, arg168_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_136 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_112: "f32[104]" = torch.ops.prims.convert_element_type.default(arg680_1, torch.float32);  arg680_1 = None
    convert_element_type_113: "f32[104]" = torch.ops.prims.convert_element_type.default(arg681_1, torch.float32);  arg681_1 = None
    add_137: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_113, 1e-05);  convert_element_type_113 = None
    sqrt_56: "f32[104]" = torch.ops.aten.sqrt.default(add_137);  add_137 = None
    reciprocal_56: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_168: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, -1);  convert_element_type_112 = None
    unsqueeze_449: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_451: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_449);  convolution_56 = unsqueeze_449 = None
    mul_169: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
    unsqueeze_453: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_170: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
    unsqueeze_454: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
    unsqueeze_455: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_138: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_53: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_138);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_53 = torch.ops.aten.split_with_sizes.default(relu_51, [104, 104, 104, 104], 1)
    getitem_216: "f32[8, 104, 14, 14]" = split_with_sizes_53[2];  split_with_sizes_53 = None
    add_139: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_53, getitem_216);  getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_57: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_139, arg171_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_139 = arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_114: "f32[104]" = torch.ops.prims.convert_element_type.default(arg683_1, torch.float32);  arg683_1 = None
    convert_element_type_115: "f32[104]" = torch.ops.prims.convert_element_type.default(arg684_1, torch.float32);  arg684_1 = None
    add_140: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
    sqrt_57: "f32[104]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_57: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_171: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
    unsqueeze_457: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
    unsqueeze_459: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
    mul_172: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
    unsqueeze_461: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_173: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
    unsqueeze_462: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
    unsqueeze_463: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_141: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_54: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_141);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_54 = torch.ops.aten.split_with_sizes.default(relu_51, [104, 104, 104, 104], 1);  relu_51 = None
    getitem_221: "f32[8, 104, 14, 14]" = split_with_sizes_54[3];  split_with_sizes_54 = None
    cat_10: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_52, relu_53, relu_54, getitem_221], 1);  relu_52 = relu_53 = relu_54 = getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_58: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_10, arg174_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_10 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_116: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg686_1, torch.float32);  arg686_1 = None
    convert_element_type_117: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg687_1, torch.float32);  arg687_1 = None
    add_142: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_117, 1e-05);  convert_element_type_117 = None
    sqrt_58: "f32[1024]" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
    reciprocal_58: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_174: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_464: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_116, -1);  convert_element_type_116 = None
    unsqueeze_465: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    unsqueeze_466: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
    unsqueeze_467: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    sub_58: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
    mul_175: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
    unsqueeze_469: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_176: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
    unsqueeze_470: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
    unsqueeze_471: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_143: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_144: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_143, relu_50);  add_143 = relu_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_55: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_59: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_55, arg177_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_118: "f32[416]" = torch.ops.prims.convert_element_type.default(arg689_1, torch.float32);  arg689_1 = None
    convert_element_type_119: "f32[416]" = torch.ops.prims.convert_element_type.default(arg690_1, torch.float32);  arg690_1 = None
    add_145: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
    sqrt_59: "f32[416]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
    reciprocal_59: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_177: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_472: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_118, -1);  convert_element_type_118 = None
    unsqueeze_473: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    unsqueeze_474: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
    unsqueeze_475: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    sub_59: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
    mul_178: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
    unsqueeze_477: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_179: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
    unsqueeze_478: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
    unsqueeze_479: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_146: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_56: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_146);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_56 = torch.ops.aten.split_with_sizes.default(relu_56, [104, 104, 104, 104], 1)
    getitem_226: "f32[8, 104, 14, 14]" = split_with_sizes_56[0];  split_with_sizes_56 = None
    convolution_60: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_226, arg180_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_226 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_120: "f32[104]" = torch.ops.prims.convert_element_type.default(arg692_1, torch.float32);  arg692_1 = None
    convert_element_type_121: "f32[104]" = torch.ops.prims.convert_element_type.default(arg693_1, torch.float32);  arg693_1 = None
    add_147: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
    sqrt_60: "f32[104]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
    reciprocal_60: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_180: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_480: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
    unsqueeze_481: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    unsqueeze_482: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
    unsqueeze_483: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    sub_60: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_481);  convolution_60 = unsqueeze_481 = None
    mul_181: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
    unsqueeze_485: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_182: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
    unsqueeze_486: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
    unsqueeze_487: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_148: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_57: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_148);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_57 = torch.ops.aten.split_with_sizes.default(relu_56, [104, 104, 104, 104], 1)
    getitem_231: "f32[8, 104, 14, 14]" = split_with_sizes_57[1];  split_with_sizes_57 = None
    add_149: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_57, getitem_231);  getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_61: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_149, arg183_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_149 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_122: "f32[104]" = torch.ops.prims.convert_element_type.default(arg695_1, torch.float32);  arg695_1 = None
    convert_element_type_123: "f32[104]" = torch.ops.prims.convert_element_type.default(arg696_1, torch.float32);  arg696_1 = None
    add_150: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_123, 1e-05);  convert_element_type_123 = None
    sqrt_61: "f32[104]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
    reciprocal_61: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_183: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_488: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_122, -1);  convert_element_type_122 = None
    unsqueeze_489: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    unsqueeze_490: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
    unsqueeze_491: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    sub_61: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
    mul_184: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
    unsqueeze_493: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_185: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
    unsqueeze_494: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
    unsqueeze_495: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_151: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_58: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_151);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_58 = torch.ops.aten.split_with_sizes.default(relu_56, [104, 104, 104, 104], 1)
    getitem_236: "f32[8, 104, 14, 14]" = split_with_sizes_58[2];  split_with_sizes_58 = None
    add_152: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_58, getitem_236);  getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_62: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_152, arg186_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_152 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_124: "f32[104]" = torch.ops.prims.convert_element_type.default(arg698_1, torch.float32);  arg698_1 = None
    convert_element_type_125: "f32[104]" = torch.ops.prims.convert_element_type.default(arg699_1, torch.float32);  arg699_1 = None
    add_153: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_125, 1e-05);  convert_element_type_125 = None
    sqrt_62: "f32[104]" = torch.ops.aten.sqrt.default(add_153);  add_153 = None
    reciprocal_62: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_186: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_496: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_124, -1);  convert_element_type_124 = None
    unsqueeze_497: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    unsqueeze_498: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_499: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    sub_62: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_497);  convolution_62 = unsqueeze_497 = None
    mul_187: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
    unsqueeze_501: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_188: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
    unsqueeze_502: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
    unsqueeze_503: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_154: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_59: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_154);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_59 = torch.ops.aten.split_with_sizes.default(relu_56, [104, 104, 104, 104], 1);  relu_56 = None
    getitem_241: "f32[8, 104, 14, 14]" = split_with_sizes_59[3];  split_with_sizes_59 = None
    cat_11: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_57, relu_58, relu_59, getitem_241], 1);  relu_57 = relu_58 = relu_59 = getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_11, arg189_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_11 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_126: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg701_1, torch.float32);  arg701_1 = None
    convert_element_type_127: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg702_1, torch.float32);  arg702_1 = None
    add_155: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
    sqrt_63: "f32[1024]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
    reciprocal_63: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_189: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_504: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
    unsqueeze_505: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    unsqueeze_506: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
    unsqueeze_507: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    sub_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_505);  convolution_63 = unsqueeze_505 = None
    mul_190: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
    unsqueeze_509: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_191: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
    unsqueeze_510: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
    unsqueeze_511: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_156: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_157: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_156, relu_55);  add_156 = relu_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_157);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_64: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_60, arg192_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_128: "f32[416]" = torch.ops.prims.convert_element_type.default(arg704_1, torch.float32);  arg704_1 = None
    convert_element_type_129: "f32[416]" = torch.ops.prims.convert_element_type.default(arg705_1, torch.float32);  arg705_1 = None
    add_158: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_129, 1e-05);  convert_element_type_129 = None
    sqrt_64: "f32[416]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
    reciprocal_64: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_192: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_512: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_128, -1);  convert_element_type_128 = None
    unsqueeze_513: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    unsqueeze_514: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_515: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    sub_64: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
    mul_193: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
    unsqueeze_517: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_194: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
    unsqueeze_518: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
    unsqueeze_519: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_159: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_61: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_159);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_61 = torch.ops.aten.split_with_sizes.default(relu_61, [104, 104, 104, 104], 1)
    getitem_246: "f32[8, 104, 14, 14]" = split_with_sizes_61[0];  split_with_sizes_61 = None
    convolution_65: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_246, arg195_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_246 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_130: "f32[104]" = torch.ops.prims.convert_element_type.default(arg707_1, torch.float32);  arg707_1 = None
    convert_element_type_131: "f32[104]" = torch.ops.prims.convert_element_type.default(arg708_1, torch.float32);  arg708_1 = None
    add_160: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_131, 1e-05);  convert_element_type_131 = None
    sqrt_65: "f32[104]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
    reciprocal_65: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
    mul_195: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
    unsqueeze_520: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_130, -1);  convert_element_type_130 = None
    unsqueeze_521: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    unsqueeze_522: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_523: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    sub_65: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
    mul_196: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
    unsqueeze_524: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
    unsqueeze_525: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_197: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
    unsqueeze_526: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
    unsqueeze_527: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_161: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_62: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_161);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_62 = torch.ops.aten.split_with_sizes.default(relu_61, [104, 104, 104, 104], 1)
    getitem_251: "f32[8, 104, 14, 14]" = split_with_sizes_62[1];  split_with_sizes_62 = None
    add_162: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_62, getitem_251);  getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_66: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_162, arg198_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_162 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_132: "f32[104]" = torch.ops.prims.convert_element_type.default(arg710_1, torch.float32);  arg710_1 = None
    convert_element_type_133: "f32[104]" = torch.ops.prims.convert_element_type.default(arg711_1, torch.float32);  arg711_1 = None
    add_163: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_133, 1e-05);  convert_element_type_133 = None
    sqrt_66: "f32[104]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
    reciprocal_66: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
    mul_198: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
    unsqueeze_528: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_132, -1);  convert_element_type_132 = None
    unsqueeze_529: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    unsqueeze_530: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
    unsqueeze_531: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    sub_66: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
    mul_199: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
    unsqueeze_532: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
    unsqueeze_533: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_200: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
    unsqueeze_534: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
    unsqueeze_535: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_164: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_63: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_164);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_63 = torch.ops.aten.split_with_sizes.default(relu_61, [104, 104, 104, 104], 1)
    getitem_256: "f32[8, 104, 14, 14]" = split_with_sizes_63[2];  split_with_sizes_63 = None
    add_165: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_63, getitem_256);  getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_67: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_165, arg201_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_165 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_134: "f32[104]" = torch.ops.prims.convert_element_type.default(arg713_1, torch.float32);  arg713_1 = None
    convert_element_type_135: "f32[104]" = torch.ops.prims.convert_element_type.default(arg714_1, torch.float32);  arg714_1 = None
    add_166: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_135, 1e-05);  convert_element_type_135 = None
    sqrt_67: "f32[104]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
    reciprocal_67: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
    mul_201: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
    unsqueeze_536: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_134, -1);  convert_element_type_134 = None
    unsqueeze_537: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    unsqueeze_538: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
    unsqueeze_539: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    sub_67: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
    mul_202: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
    unsqueeze_540: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
    unsqueeze_541: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_203: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
    unsqueeze_542: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
    unsqueeze_543: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_167: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_64: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_167);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_64 = torch.ops.aten.split_with_sizes.default(relu_61, [104, 104, 104, 104], 1);  relu_61 = None
    getitem_261: "f32[8, 104, 14, 14]" = split_with_sizes_64[3];  split_with_sizes_64 = None
    cat_12: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_62, relu_63, relu_64, getitem_261], 1);  relu_62 = relu_63 = relu_64 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_68: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_12, arg204_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_12 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_136: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg716_1, torch.float32);  arg716_1 = None
    convert_element_type_137: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg717_1, torch.float32);  arg717_1 = None
    add_168: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_137, 1e-05);  convert_element_type_137 = None
    sqrt_68: "f32[1024]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
    reciprocal_68: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
    mul_204: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
    unsqueeze_544: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_136, -1);  convert_element_type_136 = None
    unsqueeze_545: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    unsqueeze_546: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
    unsqueeze_547: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    sub_68: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
    mul_205: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
    unsqueeze_548: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
    unsqueeze_549: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_206: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
    unsqueeze_550: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
    unsqueeze_551: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_169: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_170: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_169, relu_60);  add_169 = relu_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_65: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_170);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_69: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_65, arg207_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_138: "f32[416]" = torch.ops.prims.convert_element_type.default(arg719_1, torch.float32);  arg719_1 = None
    convert_element_type_139: "f32[416]" = torch.ops.prims.convert_element_type.default(arg720_1, torch.float32);  arg720_1 = None
    add_171: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_139, 1e-05);  convert_element_type_139 = None
    sqrt_69: "f32[416]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
    reciprocal_69: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
    mul_207: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
    unsqueeze_552: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_138, -1);  convert_element_type_138 = None
    unsqueeze_553: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    unsqueeze_554: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
    unsqueeze_555: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    sub_69: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
    mul_208: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
    unsqueeze_556: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
    unsqueeze_557: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_209: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
    unsqueeze_558: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
    unsqueeze_559: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_172: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_66: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_172);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_66 = torch.ops.aten.split_with_sizes.default(relu_66, [104, 104, 104, 104], 1)
    getitem_266: "f32[8, 104, 14, 14]" = split_with_sizes_66[0];  split_with_sizes_66 = None
    convolution_70: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_266, arg210_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_266 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_140: "f32[104]" = torch.ops.prims.convert_element_type.default(arg722_1, torch.float32);  arg722_1 = None
    convert_element_type_141: "f32[104]" = torch.ops.prims.convert_element_type.default(arg723_1, torch.float32);  arg723_1 = None
    add_173: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_141, 1e-05);  convert_element_type_141 = None
    sqrt_70: "f32[104]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
    reciprocal_70: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
    mul_210: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
    unsqueeze_560: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_140, -1);  convert_element_type_140 = None
    unsqueeze_561: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    unsqueeze_562: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
    unsqueeze_563: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    sub_70: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
    mul_211: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
    unsqueeze_564: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
    unsqueeze_565: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_212: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
    unsqueeze_566: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
    unsqueeze_567: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_174: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_67: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_174);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_67 = torch.ops.aten.split_with_sizes.default(relu_66, [104, 104, 104, 104], 1)
    getitem_271: "f32[8, 104, 14, 14]" = split_with_sizes_67[1];  split_with_sizes_67 = None
    add_175: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_67, getitem_271);  getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_71: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_175, arg213_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_175 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_142: "f32[104]" = torch.ops.prims.convert_element_type.default(arg725_1, torch.float32);  arg725_1 = None
    convert_element_type_143: "f32[104]" = torch.ops.prims.convert_element_type.default(arg726_1, torch.float32);  arg726_1 = None
    add_176: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_143, 1e-05);  convert_element_type_143 = None
    sqrt_71: "f32[104]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
    reciprocal_71: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
    mul_213: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
    unsqueeze_568: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_142, -1);  convert_element_type_142 = None
    unsqueeze_569: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    unsqueeze_570: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
    unsqueeze_571: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    sub_71: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
    mul_214: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
    unsqueeze_572: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
    unsqueeze_573: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_215: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
    unsqueeze_574: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
    unsqueeze_575: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_177: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_68: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_177);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_68 = torch.ops.aten.split_with_sizes.default(relu_66, [104, 104, 104, 104], 1)
    getitem_276: "f32[8, 104, 14, 14]" = split_with_sizes_68[2];  split_with_sizes_68 = None
    add_178: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_68, getitem_276);  getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_72: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_178, arg216_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_178 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_144: "f32[104]" = torch.ops.prims.convert_element_type.default(arg728_1, torch.float32);  arg728_1 = None
    convert_element_type_145: "f32[104]" = torch.ops.prims.convert_element_type.default(arg729_1, torch.float32);  arg729_1 = None
    add_179: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_145, 1e-05);  convert_element_type_145 = None
    sqrt_72: "f32[104]" = torch.ops.aten.sqrt.default(add_179);  add_179 = None
    reciprocal_72: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
    mul_216: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
    unsqueeze_576: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_144, -1);  convert_element_type_144 = None
    unsqueeze_577: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    unsqueeze_578: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
    unsqueeze_579: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    sub_72: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
    mul_217: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
    unsqueeze_580: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
    unsqueeze_581: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_218: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
    unsqueeze_582: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
    unsqueeze_583: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_180: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_69: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_180);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_69 = torch.ops.aten.split_with_sizes.default(relu_66, [104, 104, 104, 104], 1);  relu_66 = None
    getitem_281: "f32[8, 104, 14, 14]" = split_with_sizes_69[3];  split_with_sizes_69 = None
    cat_13: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_67, relu_68, relu_69, getitem_281], 1);  relu_67 = relu_68 = relu_69 = getitem_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_73: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_13, arg219_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_13 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_146: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg731_1, torch.float32);  arg731_1 = None
    convert_element_type_147: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg732_1, torch.float32);  arg732_1 = None
    add_181: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_147, 1e-05);  convert_element_type_147 = None
    sqrt_73: "f32[1024]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
    reciprocal_73: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
    mul_219: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
    unsqueeze_584: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_146, -1);  convert_element_type_146 = None
    unsqueeze_585: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    unsqueeze_586: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
    unsqueeze_587: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    sub_73: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
    mul_220: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
    unsqueeze_588: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
    unsqueeze_589: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_221: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
    unsqueeze_590: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
    unsqueeze_591: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_182: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_183: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_182, relu_65);  add_182 = relu_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_70: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_183);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_74: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_70, arg222_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_148: "f32[416]" = torch.ops.prims.convert_element_type.default(arg734_1, torch.float32);  arg734_1 = None
    convert_element_type_149: "f32[416]" = torch.ops.prims.convert_element_type.default(arg735_1, torch.float32);  arg735_1 = None
    add_184: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_149, 1e-05);  convert_element_type_149 = None
    sqrt_74: "f32[416]" = torch.ops.aten.sqrt.default(add_184);  add_184 = None
    reciprocal_74: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
    mul_222: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
    unsqueeze_592: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_148, -1);  convert_element_type_148 = None
    unsqueeze_593: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    unsqueeze_594: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
    unsqueeze_595: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    sub_74: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
    mul_223: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
    unsqueeze_596: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
    unsqueeze_597: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_224: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
    unsqueeze_598: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
    unsqueeze_599: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_185: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_71: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_185);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_71 = torch.ops.aten.split_with_sizes.default(relu_71, [104, 104, 104, 104], 1)
    getitem_286: "f32[8, 104, 14, 14]" = split_with_sizes_71[0];  split_with_sizes_71 = None
    convolution_75: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_286, arg225_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_286 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_150: "f32[104]" = torch.ops.prims.convert_element_type.default(arg737_1, torch.float32);  arg737_1 = None
    convert_element_type_151: "f32[104]" = torch.ops.prims.convert_element_type.default(arg738_1, torch.float32);  arg738_1 = None
    add_186: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_151, 1e-05);  convert_element_type_151 = None
    sqrt_75: "f32[104]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
    reciprocal_75: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
    mul_225: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
    unsqueeze_600: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_150, -1);  convert_element_type_150 = None
    unsqueeze_601: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    unsqueeze_602: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
    unsqueeze_603: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    sub_75: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
    mul_226: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
    unsqueeze_604: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
    unsqueeze_605: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_227: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
    unsqueeze_606: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
    unsqueeze_607: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_187: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_72: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_187);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_72 = torch.ops.aten.split_with_sizes.default(relu_71, [104, 104, 104, 104], 1)
    getitem_291: "f32[8, 104, 14, 14]" = split_with_sizes_72[1];  split_with_sizes_72 = None
    add_188: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_72, getitem_291);  getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_76: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_188, arg228_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_188 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_152: "f32[104]" = torch.ops.prims.convert_element_type.default(arg740_1, torch.float32);  arg740_1 = None
    convert_element_type_153: "f32[104]" = torch.ops.prims.convert_element_type.default(arg741_1, torch.float32);  arg741_1 = None
    add_189: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_153, 1e-05);  convert_element_type_153 = None
    sqrt_76: "f32[104]" = torch.ops.aten.sqrt.default(add_189);  add_189 = None
    reciprocal_76: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
    mul_228: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
    unsqueeze_608: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_152, -1);  convert_element_type_152 = None
    unsqueeze_609: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    unsqueeze_610: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
    unsqueeze_611: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    sub_76: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
    mul_229: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
    unsqueeze_612: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
    unsqueeze_613: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_230: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
    unsqueeze_614: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
    unsqueeze_615: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_190: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_73: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_190);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_73 = torch.ops.aten.split_with_sizes.default(relu_71, [104, 104, 104, 104], 1)
    getitem_296: "f32[8, 104, 14, 14]" = split_with_sizes_73[2];  split_with_sizes_73 = None
    add_191: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_73, getitem_296);  getitem_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_77: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_191, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_191 = arg231_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_154: "f32[104]" = torch.ops.prims.convert_element_type.default(arg743_1, torch.float32);  arg743_1 = None
    convert_element_type_155: "f32[104]" = torch.ops.prims.convert_element_type.default(arg744_1, torch.float32);  arg744_1 = None
    add_192: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_155, 1e-05);  convert_element_type_155 = None
    sqrt_77: "f32[104]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
    reciprocal_77: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
    mul_231: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
    unsqueeze_616: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_154, -1);  convert_element_type_154 = None
    unsqueeze_617: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    unsqueeze_618: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
    unsqueeze_619: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    sub_77: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
    mul_232: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
    unsqueeze_620: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
    unsqueeze_621: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_233: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
    unsqueeze_622: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
    unsqueeze_623: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_193: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_74: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_193);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_74 = torch.ops.aten.split_with_sizes.default(relu_71, [104, 104, 104, 104], 1);  relu_71 = None
    getitem_301: "f32[8, 104, 14, 14]" = split_with_sizes_74[3];  split_with_sizes_74 = None
    cat_14: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_72, relu_73, relu_74, getitem_301], 1);  relu_72 = relu_73 = relu_74 = getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_78: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_14, arg234_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_14 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_156: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg746_1, torch.float32);  arg746_1 = None
    convert_element_type_157: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg747_1, torch.float32);  arg747_1 = None
    add_194: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_157, 1e-05);  convert_element_type_157 = None
    sqrt_78: "f32[1024]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
    reciprocal_78: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
    mul_234: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
    unsqueeze_624: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_156, -1);  convert_element_type_156 = None
    unsqueeze_625: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    unsqueeze_626: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
    unsqueeze_627: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    sub_78: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
    mul_235: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
    unsqueeze_628: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
    unsqueeze_629: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_236: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
    unsqueeze_630: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
    unsqueeze_631: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_195: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_196: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_195, relu_70);  add_195 = relu_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_196);  add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_79: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_75, arg237_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_158: "f32[416]" = torch.ops.prims.convert_element_type.default(arg749_1, torch.float32);  arg749_1 = None
    convert_element_type_159: "f32[416]" = torch.ops.prims.convert_element_type.default(arg750_1, torch.float32);  arg750_1 = None
    add_197: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_159, 1e-05);  convert_element_type_159 = None
    sqrt_79: "f32[416]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
    reciprocal_79: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
    mul_237: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
    unsqueeze_632: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_158, -1);  convert_element_type_158 = None
    unsqueeze_633: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    unsqueeze_634: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
    unsqueeze_635: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    sub_79: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
    mul_238: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
    unsqueeze_636: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
    unsqueeze_637: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_239: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
    unsqueeze_638: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
    unsqueeze_639: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_198: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_76: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_198);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_76 = torch.ops.aten.split_with_sizes.default(relu_76, [104, 104, 104, 104], 1)
    getitem_306: "f32[8, 104, 14, 14]" = split_with_sizes_76[0];  split_with_sizes_76 = None
    convolution_80: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_306, arg240_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_306 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_160: "f32[104]" = torch.ops.prims.convert_element_type.default(arg752_1, torch.float32);  arg752_1 = None
    convert_element_type_161: "f32[104]" = torch.ops.prims.convert_element_type.default(arg753_1, torch.float32);  arg753_1 = None
    add_199: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_161, 1e-05);  convert_element_type_161 = None
    sqrt_80: "f32[104]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
    reciprocal_80: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
    mul_240: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
    unsqueeze_640: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_160, -1);  convert_element_type_160 = None
    unsqueeze_641: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
    unsqueeze_642: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
    unsqueeze_643: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
    sub_80: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
    mul_241: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
    unsqueeze_644: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
    unsqueeze_645: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
    mul_242: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
    unsqueeze_646: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
    unsqueeze_647: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    add_200: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_77: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_200);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_77 = torch.ops.aten.split_with_sizes.default(relu_76, [104, 104, 104, 104], 1)
    getitem_311: "f32[8, 104, 14, 14]" = split_with_sizes_77[1];  split_with_sizes_77 = None
    add_201: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_77, getitem_311);  getitem_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_81: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_201, arg243_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_201 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_162: "f32[104]" = torch.ops.prims.convert_element_type.default(arg755_1, torch.float32);  arg755_1 = None
    convert_element_type_163: "f32[104]" = torch.ops.prims.convert_element_type.default(arg756_1, torch.float32);  arg756_1 = None
    add_202: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_163, 1e-05);  convert_element_type_163 = None
    sqrt_81: "f32[104]" = torch.ops.aten.sqrt.default(add_202);  add_202 = None
    reciprocal_81: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
    mul_243: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
    unsqueeze_648: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_162, -1);  convert_element_type_162 = None
    unsqueeze_649: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    unsqueeze_650: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
    unsqueeze_651: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
    sub_81: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
    mul_244: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
    unsqueeze_652: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
    unsqueeze_653: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
    mul_245: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
    unsqueeze_654: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
    unsqueeze_655: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
    add_203: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_78: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_203);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_78 = torch.ops.aten.split_with_sizes.default(relu_76, [104, 104, 104, 104], 1)
    getitem_316: "f32[8, 104, 14, 14]" = split_with_sizes_78[2];  split_with_sizes_78 = None
    add_204: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_78, getitem_316);  getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_82: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_204, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_204 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_164: "f32[104]" = torch.ops.prims.convert_element_type.default(arg758_1, torch.float32);  arg758_1 = None
    convert_element_type_165: "f32[104]" = torch.ops.prims.convert_element_type.default(arg759_1, torch.float32);  arg759_1 = None
    add_205: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_165, 1e-05);  convert_element_type_165 = None
    sqrt_82: "f32[104]" = torch.ops.aten.sqrt.default(add_205);  add_205 = None
    reciprocal_82: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
    mul_246: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
    unsqueeze_656: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_164, -1);  convert_element_type_164 = None
    unsqueeze_657: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    unsqueeze_658: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
    unsqueeze_659: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    sub_82: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_657);  convolution_82 = unsqueeze_657 = None
    mul_247: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
    unsqueeze_660: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
    unsqueeze_661: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
    mul_248: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
    unsqueeze_662: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
    unsqueeze_663: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
    add_206: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_79: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_206);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_79 = torch.ops.aten.split_with_sizes.default(relu_76, [104, 104, 104, 104], 1);  relu_76 = None
    getitem_321: "f32[8, 104, 14, 14]" = split_with_sizes_79[3];  split_with_sizes_79 = None
    cat_15: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_77, relu_78, relu_79, getitem_321], 1);  relu_77 = relu_78 = relu_79 = getitem_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_83: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_15, arg249_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_15 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_166: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg761_1, torch.float32);  arg761_1 = None
    convert_element_type_167: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg762_1, torch.float32);  arg762_1 = None
    add_207: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_167, 1e-05);  convert_element_type_167 = None
    sqrt_83: "f32[1024]" = torch.ops.aten.sqrt.default(add_207);  add_207 = None
    reciprocal_83: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
    mul_249: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
    unsqueeze_664: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_166, -1);  convert_element_type_166 = None
    unsqueeze_665: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
    unsqueeze_666: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
    unsqueeze_667: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    sub_83: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_665);  convolution_83 = unsqueeze_665 = None
    mul_250: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
    unsqueeze_668: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
    unsqueeze_669: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    mul_251: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
    unsqueeze_670: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
    unsqueeze_671: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
    add_208: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_209: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_208, relu_75);  add_208 = relu_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_80: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_209);  add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_84: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_80, arg252_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_168: "f32[416]" = torch.ops.prims.convert_element_type.default(arg764_1, torch.float32);  arg764_1 = None
    convert_element_type_169: "f32[416]" = torch.ops.prims.convert_element_type.default(arg765_1, torch.float32);  arg765_1 = None
    add_210: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_169, 1e-05);  convert_element_type_169 = None
    sqrt_84: "f32[416]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
    reciprocal_84: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
    mul_252: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
    unsqueeze_672: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_168, -1);  convert_element_type_168 = None
    unsqueeze_673: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
    unsqueeze_674: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
    unsqueeze_675: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
    sub_84: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_673);  convolution_84 = unsqueeze_673 = None
    mul_253: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
    unsqueeze_676: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
    unsqueeze_677: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    mul_254: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
    unsqueeze_678: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
    unsqueeze_679: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    add_211: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_81: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_81 = torch.ops.aten.split_with_sizes.default(relu_81, [104, 104, 104, 104], 1)
    getitem_326: "f32[8, 104, 14, 14]" = split_with_sizes_81[0];  split_with_sizes_81 = None
    convolution_85: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_326, arg255_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_326 = arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_170: "f32[104]" = torch.ops.prims.convert_element_type.default(arg767_1, torch.float32);  arg767_1 = None
    convert_element_type_171: "f32[104]" = torch.ops.prims.convert_element_type.default(arg768_1, torch.float32);  arg768_1 = None
    add_212: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_171, 1e-05);  convert_element_type_171 = None
    sqrt_85: "f32[104]" = torch.ops.aten.sqrt.default(add_212);  add_212 = None
    reciprocal_85: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
    mul_255: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
    unsqueeze_680: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_170, -1);  convert_element_type_170 = None
    unsqueeze_681: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
    unsqueeze_682: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
    unsqueeze_683: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
    sub_85: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_681);  convolution_85 = unsqueeze_681 = None
    mul_256: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
    unsqueeze_684: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
    unsqueeze_685: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
    mul_257: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
    unsqueeze_686: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
    unsqueeze_687: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
    add_213: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_82: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_213);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_82 = torch.ops.aten.split_with_sizes.default(relu_81, [104, 104, 104, 104], 1)
    getitem_331: "f32[8, 104, 14, 14]" = split_with_sizes_82[1];  split_with_sizes_82 = None
    add_214: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_82, getitem_331);  getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_86: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_214, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_214 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_172: "f32[104]" = torch.ops.prims.convert_element_type.default(arg770_1, torch.float32);  arg770_1 = None
    convert_element_type_173: "f32[104]" = torch.ops.prims.convert_element_type.default(arg771_1, torch.float32);  arg771_1 = None
    add_215: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_173, 1e-05);  convert_element_type_173 = None
    sqrt_86: "f32[104]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
    reciprocal_86: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
    mul_258: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
    unsqueeze_688: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_172, -1);  convert_element_type_172 = None
    unsqueeze_689: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
    unsqueeze_690: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
    unsqueeze_691: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
    sub_86: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_689);  convolution_86 = unsqueeze_689 = None
    mul_259: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
    unsqueeze_692: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
    unsqueeze_693: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
    mul_260: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
    unsqueeze_694: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
    unsqueeze_695: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
    add_216: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_83: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_83 = torch.ops.aten.split_with_sizes.default(relu_81, [104, 104, 104, 104], 1)
    getitem_336: "f32[8, 104, 14, 14]" = split_with_sizes_83[2];  split_with_sizes_83 = None
    add_217: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_83, getitem_336);  getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_87: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_217, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_217 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_174: "f32[104]" = torch.ops.prims.convert_element_type.default(arg773_1, torch.float32);  arg773_1 = None
    convert_element_type_175: "f32[104]" = torch.ops.prims.convert_element_type.default(arg774_1, torch.float32);  arg774_1 = None
    add_218: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_175, 1e-05);  convert_element_type_175 = None
    sqrt_87: "f32[104]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
    reciprocal_87: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
    mul_261: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
    unsqueeze_696: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_174, -1);  convert_element_type_174 = None
    unsqueeze_697: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
    unsqueeze_698: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
    unsqueeze_699: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
    sub_87: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_697);  convolution_87 = unsqueeze_697 = None
    mul_262: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
    unsqueeze_700: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
    unsqueeze_701: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
    mul_263: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
    unsqueeze_702: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
    unsqueeze_703: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
    add_219: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_84: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_219);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_84 = torch.ops.aten.split_with_sizes.default(relu_81, [104, 104, 104, 104], 1);  relu_81 = None
    getitem_341: "f32[8, 104, 14, 14]" = split_with_sizes_84[3];  split_with_sizes_84 = None
    cat_16: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_82, relu_83, relu_84, getitem_341], 1);  relu_82 = relu_83 = relu_84 = getitem_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_88: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_16, arg264_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_16 = arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_176: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg776_1, torch.float32);  arg776_1 = None
    convert_element_type_177: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg777_1, torch.float32);  arg777_1 = None
    add_220: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_177, 1e-05);  convert_element_type_177 = None
    sqrt_88: "f32[1024]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
    reciprocal_88: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
    mul_264: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
    unsqueeze_704: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_176, -1);  convert_element_type_176 = None
    unsqueeze_705: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
    unsqueeze_706: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
    unsqueeze_707: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
    sub_88: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_705);  convolution_88 = unsqueeze_705 = None
    mul_265: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
    unsqueeze_708: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
    unsqueeze_709: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
    mul_266: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
    unsqueeze_710: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
    unsqueeze_711: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
    add_221: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_222: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_221, relu_80);  add_221 = relu_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_85: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_222);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_89: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_85, arg267_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_178: "f32[416]" = torch.ops.prims.convert_element_type.default(arg779_1, torch.float32);  arg779_1 = None
    convert_element_type_179: "f32[416]" = torch.ops.prims.convert_element_type.default(arg780_1, torch.float32);  arg780_1 = None
    add_223: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_179, 1e-05);  convert_element_type_179 = None
    sqrt_89: "f32[416]" = torch.ops.aten.sqrt.default(add_223);  add_223 = None
    reciprocal_89: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
    mul_267: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
    unsqueeze_712: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_178, -1);  convert_element_type_178 = None
    unsqueeze_713: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
    unsqueeze_714: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
    unsqueeze_715: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
    sub_89: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_713);  convolution_89 = unsqueeze_713 = None
    mul_268: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
    unsqueeze_716: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
    unsqueeze_717: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
    mul_269: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
    unsqueeze_718: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
    unsqueeze_719: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
    add_224: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_86: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_224);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_86 = torch.ops.aten.split_with_sizes.default(relu_86, [104, 104, 104, 104], 1)
    getitem_346: "f32[8, 104, 14, 14]" = split_with_sizes_86[0];  split_with_sizes_86 = None
    convolution_90: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_346, arg270_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_346 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_180: "f32[104]" = torch.ops.prims.convert_element_type.default(arg782_1, torch.float32);  arg782_1 = None
    convert_element_type_181: "f32[104]" = torch.ops.prims.convert_element_type.default(arg783_1, torch.float32);  arg783_1 = None
    add_225: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_181, 1e-05);  convert_element_type_181 = None
    sqrt_90: "f32[104]" = torch.ops.aten.sqrt.default(add_225);  add_225 = None
    reciprocal_90: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
    mul_270: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
    unsqueeze_720: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_180, -1);  convert_element_type_180 = None
    unsqueeze_721: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
    unsqueeze_722: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
    unsqueeze_723: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
    sub_90: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_721);  convolution_90 = unsqueeze_721 = None
    mul_271: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
    unsqueeze_724: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
    unsqueeze_725: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
    mul_272: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
    unsqueeze_726: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
    unsqueeze_727: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
    add_226: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_87: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_87 = torch.ops.aten.split_with_sizes.default(relu_86, [104, 104, 104, 104], 1)
    getitem_351: "f32[8, 104, 14, 14]" = split_with_sizes_87[1];  split_with_sizes_87 = None
    add_227: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_87, getitem_351);  getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_91: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_227, arg273_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_227 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_182: "f32[104]" = torch.ops.prims.convert_element_type.default(arg785_1, torch.float32);  arg785_1 = None
    convert_element_type_183: "f32[104]" = torch.ops.prims.convert_element_type.default(arg786_1, torch.float32);  arg786_1 = None
    add_228: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_183, 1e-05);  convert_element_type_183 = None
    sqrt_91: "f32[104]" = torch.ops.aten.sqrt.default(add_228);  add_228 = None
    reciprocal_91: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
    mul_273: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
    unsqueeze_728: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_182, -1);  convert_element_type_182 = None
    unsqueeze_729: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
    unsqueeze_730: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
    unsqueeze_731: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
    sub_91: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_729);  convolution_91 = unsqueeze_729 = None
    mul_274: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
    unsqueeze_732: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
    unsqueeze_733: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
    mul_275: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
    unsqueeze_734: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
    unsqueeze_735: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
    add_229: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_88: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_229);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_88 = torch.ops.aten.split_with_sizes.default(relu_86, [104, 104, 104, 104], 1)
    getitem_356: "f32[8, 104, 14, 14]" = split_with_sizes_88[2];  split_with_sizes_88 = None
    add_230: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_88, getitem_356);  getitem_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_92: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_230, arg276_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_230 = arg276_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_184: "f32[104]" = torch.ops.prims.convert_element_type.default(arg788_1, torch.float32);  arg788_1 = None
    convert_element_type_185: "f32[104]" = torch.ops.prims.convert_element_type.default(arg789_1, torch.float32);  arg789_1 = None
    add_231: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_185, 1e-05);  convert_element_type_185 = None
    sqrt_92: "f32[104]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
    reciprocal_92: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
    mul_276: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
    unsqueeze_736: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_184, -1);  convert_element_type_184 = None
    unsqueeze_737: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
    unsqueeze_738: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
    unsqueeze_739: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
    sub_92: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_737);  convolution_92 = unsqueeze_737 = None
    mul_277: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
    unsqueeze_740: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
    unsqueeze_741: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
    mul_278: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
    unsqueeze_742: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
    unsqueeze_743: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
    add_232: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_89: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_232);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_89 = torch.ops.aten.split_with_sizes.default(relu_86, [104, 104, 104, 104], 1);  relu_86 = None
    getitem_361: "f32[8, 104, 14, 14]" = split_with_sizes_89[3];  split_with_sizes_89 = None
    cat_17: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_87, relu_88, relu_89, getitem_361], 1);  relu_87 = relu_88 = relu_89 = getitem_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_17, arg279_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_17 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_186: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg791_1, torch.float32);  arg791_1 = None
    convert_element_type_187: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg792_1, torch.float32);  arg792_1 = None
    add_233: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_187, 1e-05);  convert_element_type_187 = None
    sqrt_93: "f32[1024]" = torch.ops.aten.sqrt.default(add_233);  add_233 = None
    reciprocal_93: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
    mul_279: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
    unsqueeze_744: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_186, -1);  convert_element_type_186 = None
    unsqueeze_745: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
    unsqueeze_746: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
    unsqueeze_747: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
    sub_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_745);  convolution_93 = unsqueeze_745 = None
    mul_280: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
    unsqueeze_748: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
    unsqueeze_749: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
    mul_281: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
    unsqueeze_750: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
    unsqueeze_751: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
    add_234: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_235: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_234, relu_85);  add_234 = relu_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_90: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_235);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_94: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_90, arg282_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_188: "f32[416]" = torch.ops.prims.convert_element_type.default(arg794_1, torch.float32);  arg794_1 = None
    convert_element_type_189: "f32[416]" = torch.ops.prims.convert_element_type.default(arg795_1, torch.float32);  arg795_1 = None
    add_236: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_189, 1e-05);  convert_element_type_189 = None
    sqrt_94: "f32[416]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
    reciprocal_94: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
    mul_282: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
    unsqueeze_752: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_188, -1);  convert_element_type_188 = None
    unsqueeze_753: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
    unsqueeze_754: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
    unsqueeze_755: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
    sub_94: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_753);  convolution_94 = unsqueeze_753 = None
    mul_283: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
    unsqueeze_756: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
    unsqueeze_757: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
    mul_284: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
    unsqueeze_758: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
    unsqueeze_759: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
    add_237: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_91: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_91 = torch.ops.aten.split_with_sizes.default(relu_91, [104, 104, 104, 104], 1)
    getitem_366: "f32[8, 104, 14, 14]" = split_with_sizes_91[0];  split_with_sizes_91 = None
    convolution_95: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_366, arg285_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_366 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_190: "f32[104]" = torch.ops.prims.convert_element_type.default(arg797_1, torch.float32);  arg797_1 = None
    convert_element_type_191: "f32[104]" = torch.ops.prims.convert_element_type.default(arg798_1, torch.float32);  arg798_1 = None
    add_238: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_191, 1e-05);  convert_element_type_191 = None
    sqrt_95: "f32[104]" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
    reciprocal_95: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
    mul_285: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
    unsqueeze_760: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_190, -1);  convert_element_type_190 = None
    unsqueeze_761: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
    unsqueeze_762: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
    unsqueeze_763: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
    sub_95: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_761);  convolution_95 = unsqueeze_761 = None
    mul_286: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
    unsqueeze_764: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg286_1, -1);  arg286_1 = None
    unsqueeze_765: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
    mul_287: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
    unsqueeze_766: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
    unsqueeze_767: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
    add_239: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_92: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_239);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_92 = torch.ops.aten.split_with_sizes.default(relu_91, [104, 104, 104, 104], 1)
    getitem_371: "f32[8, 104, 14, 14]" = split_with_sizes_92[1];  split_with_sizes_92 = None
    add_240: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_92, getitem_371);  getitem_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_96: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_240, arg288_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_240 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_192: "f32[104]" = torch.ops.prims.convert_element_type.default(arg800_1, torch.float32);  arg800_1 = None
    convert_element_type_193: "f32[104]" = torch.ops.prims.convert_element_type.default(arg801_1, torch.float32);  arg801_1 = None
    add_241: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_193, 1e-05);  convert_element_type_193 = None
    sqrt_96: "f32[104]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
    reciprocal_96: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
    mul_288: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
    unsqueeze_768: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_192, -1);  convert_element_type_192 = None
    unsqueeze_769: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
    unsqueeze_770: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
    unsqueeze_771: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
    sub_96: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_769);  convolution_96 = unsqueeze_769 = None
    mul_289: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
    unsqueeze_772: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
    unsqueeze_773: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
    mul_290: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
    unsqueeze_774: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
    unsqueeze_775: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
    add_242: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_93: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_242);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_93 = torch.ops.aten.split_with_sizes.default(relu_91, [104, 104, 104, 104], 1)
    getitem_376: "f32[8, 104, 14, 14]" = split_with_sizes_93[2];  split_with_sizes_93 = None
    add_243: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_93, getitem_376);  getitem_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_97: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_243, arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_243 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_194: "f32[104]" = torch.ops.prims.convert_element_type.default(arg803_1, torch.float32);  arg803_1 = None
    convert_element_type_195: "f32[104]" = torch.ops.prims.convert_element_type.default(arg804_1, torch.float32);  arg804_1 = None
    add_244: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_195, 1e-05);  convert_element_type_195 = None
    sqrt_97: "f32[104]" = torch.ops.aten.sqrt.default(add_244);  add_244 = None
    reciprocal_97: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
    mul_291: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
    unsqueeze_776: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_194, -1);  convert_element_type_194 = None
    unsqueeze_777: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
    unsqueeze_778: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
    unsqueeze_779: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
    sub_97: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_777);  convolution_97 = unsqueeze_777 = None
    mul_292: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
    unsqueeze_780: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
    unsqueeze_781: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
    mul_293: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
    unsqueeze_782: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
    unsqueeze_783: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
    add_245: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_94: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_245);  add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_94 = torch.ops.aten.split_with_sizes.default(relu_91, [104, 104, 104, 104], 1);  relu_91 = None
    getitem_381: "f32[8, 104, 14, 14]" = split_with_sizes_94[3];  split_with_sizes_94 = None
    cat_18: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_92, relu_93, relu_94, getitem_381], 1);  relu_92 = relu_93 = relu_94 = getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_98: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_18, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_18 = arg294_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_196: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg806_1, torch.float32);  arg806_1 = None
    convert_element_type_197: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg807_1, torch.float32);  arg807_1 = None
    add_246: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_197, 1e-05);  convert_element_type_197 = None
    sqrt_98: "f32[1024]" = torch.ops.aten.sqrt.default(add_246);  add_246 = None
    reciprocal_98: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
    mul_294: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
    unsqueeze_784: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_196, -1);  convert_element_type_196 = None
    unsqueeze_785: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
    unsqueeze_786: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
    unsqueeze_787: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
    sub_98: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_785);  convolution_98 = unsqueeze_785 = None
    mul_295: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
    unsqueeze_788: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
    unsqueeze_789: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
    mul_296: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
    unsqueeze_790: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
    unsqueeze_791: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
    add_247: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_248: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_247, relu_90);  add_247 = relu_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_95: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_248);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_99: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_95, arg297_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_198: "f32[416]" = torch.ops.prims.convert_element_type.default(arg809_1, torch.float32);  arg809_1 = None
    convert_element_type_199: "f32[416]" = torch.ops.prims.convert_element_type.default(arg810_1, torch.float32);  arg810_1 = None
    add_249: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_199, 1e-05);  convert_element_type_199 = None
    sqrt_99: "f32[416]" = torch.ops.aten.sqrt.default(add_249);  add_249 = None
    reciprocal_99: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
    mul_297: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
    unsqueeze_792: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_198, -1);  convert_element_type_198 = None
    unsqueeze_793: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
    unsqueeze_794: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
    unsqueeze_795: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
    sub_99: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_793);  convolution_99 = unsqueeze_793 = None
    mul_298: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
    unsqueeze_796: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
    unsqueeze_797: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
    mul_299: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
    unsqueeze_798: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
    unsqueeze_799: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
    add_250: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_96: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_250);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_96 = torch.ops.aten.split_with_sizes.default(relu_96, [104, 104, 104, 104], 1)
    getitem_386: "f32[8, 104, 14, 14]" = split_with_sizes_96[0];  split_with_sizes_96 = None
    convolution_100: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_386, arg300_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_386 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_200: "f32[104]" = torch.ops.prims.convert_element_type.default(arg812_1, torch.float32);  arg812_1 = None
    convert_element_type_201: "f32[104]" = torch.ops.prims.convert_element_type.default(arg813_1, torch.float32);  arg813_1 = None
    add_251: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_201, 1e-05);  convert_element_type_201 = None
    sqrt_100: "f32[104]" = torch.ops.aten.sqrt.default(add_251);  add_251 = None
    reciprocal_100: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
    mul_300: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
    unsqueeze_800: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_200, -1);  convert_element_type_200 = None
    unsqueeze_801: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
    unsqueeze_802: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
    unsqueeze_803: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
    sub_100: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_801);  convolution_100 = unsqueeze_801 = None
    mul_301: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
    unsqueeze_804: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
    unsqueeze_805: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
    mul_302: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
    unsqueeze_806: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
    unsqueeze_807: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
    add_252: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_97: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_252);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_97 = torch.ops.aten.split_with_sizes.default(relu_96, [104, 104, 104, 104], 1)
    getitem_391: "f32[8, 104, 14, 14]" = split_with_sizes_97[1];  split_with_sizes_97 = None
    add_253: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_97, getitem_391);  getitem_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_101: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_253, arg303_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_253 = arg303_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_202: "f32[104]" = torch.ops.prims.convert_element_type.default(arg815_1, torch.float32);  arg815_1 = None
    convert_element_type_203: "f32[104]" = torch.ops.prims.convert_element_type.default(arg816_1, torch.float32);  arg816_1 = None
    add_254: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_203, 1e-05);  convert_element_type_203 = None
    sqrt_101: "f32[104]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
    reciprocal_101: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
    mul_303: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
    unsqueeze_808: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_202, -1);  convert_element_type_202 = None
    unsqueeze_809: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
    unsqueeze_810: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
    unsqueeze_811: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
    sub_101: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_809);  convolution_101 = unsqueeze_809 = None
    mul_304: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
    unsqueeze_812: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
    unsqueeze_813: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
    mul_305: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
    unsqueeze_814: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
    unsqueeze_815: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
    add_255: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_98: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_255);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_98 = torch.ops.aten.split_with_sizes.default(relu_96, [104, 104, 104, 104], 1)
    getitem_396: "f32[8, 104, 14, 14]" = split_with_sizes_98[2];  split_with_sizes_98 = None
    add_256: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_98, getitem_396);  getitem_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_102: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_256, arg306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_256 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_204: "f32[104]" = torch.ops.prims.convert_element_type.default(arg818_1, torch.float32);  arg818_1 = None
    convert_element_type_205: "f32[104]" = torch.ops.prims.convert_element_type.default(arg819_1, torch.float32);  arg819_1 = None
    add_257: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_205, 1e-05);  convert_element_type_205 = None
    sqrt_102: "f32[104]" = torch.ops.aten.sqrt.default(add_257);  add_257 = None
    reciprocal_102: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
    mul_306: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
    unsqueeze_816: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_204, -1);  convert_element_type_204 = None
    unsqueeze_817: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
    unsqueeze_818: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
    unsqueeze_819: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
    sub_102: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_817);  convolution_102 = unsqueeze_817 = None
    mul_307: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
    unsqueeze_820: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
    unsqueeze_821: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
    mul_308: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
    unsqueeze_822: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
    unsqueeze_823: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
    add_258: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_99: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_258);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_99 = torch.ops.aten.split_with_sizes.default(relu_96, [104, 104, 104, 104], 1);  relu_96 = None
    getitem_401: "f32[8, 104, 14, 14]" = split_with_sizes_99[3];  split_with_sizes_99 = None
    cat_19: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_97, relu_98, relu_99, getitem_401], 1);  relu_97 = relu_98 = relu_99 = getitem_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_103: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_19, arg309_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_19 = arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_206: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg821_1, torch.float32);  arg821_1 = None
    convert_element_type_207: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg822_1, torch.float32);  arg822_1 = None
    add_259: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_207, 1e-05);  convert_element_type_207 = None
    sqrt_103: "f32[1024]" = torch.ops.aten.sqrt.default(add_259);  add_259 = None
    reciprocal_103: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
    mul_309: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
    unsqueeze_824: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_206, -1);  convert_element_type_206 = None
    unsqueeze_825: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
    unsqueeze_826: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
    unsqueeze_827: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
    sub_103: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_825);  convolution_103 = unsqueeze_825 = None
    mul_310: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
    unsqueeze_828: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
    unsqueeze_829: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
    mul_311: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
    unsqueeze_830: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
    unsqueeze_831: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
    add_260: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_261: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_260, relu_95);  add_260 = relu_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_100: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_261);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_104: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_100, arg312_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_208: "f32[416]" = torch.ops.prims.convert_element_type.default(arg824_1, torch.float32);  arg824_1 = None
    convert_element_type_209: "f32[416]" = torch.ops.prims.convert_element_type.default(arg825_1, torch.float32);  arg825_1 = None
    add_262: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_209, 1e-05);  convert_element_type_209 = None
    sqrt_104: "f32[416]" = torch.ops.aten.sqrt.default(add_262);  add_262 = None
    reciprocal_104: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
    mul_312: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
    unsqueeze_832: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_208, -1);  convert_element_type_208 = None
    unsqueeze_833: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
    unsqueeze_834: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
    unsqueeze_835: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
    sub_104: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_833);  convolution_104 = unsqueeze_833 = None
    mul_313: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
    unsqueeze_836: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
    unsqueeze_837: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
    mul_314: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
    unsqueeze_838: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
    unsqueeze_839: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
    add_263: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_101: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_263);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_101 = torch.ops.aten.split_with_sizes.default(relu_101, [104, 104, 104, 104], 1)
    getitem_406: "f32[8, 104, 14, 14]" = split_with_sizes_101[0];  split_with_sizes_101 = None
    convolution_105: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_406, arg315_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_406 = arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_210: "f32[104]" = torch.ops.prims.convert_element_type.default(arg827_1, torch.float32);  arg827_1 = None
    convert_element_type_211: "f32[104]" = torch.ops.prims.convert_element_type.default(arg828_1, torch.float32);  arg828_1 = None
    add_264: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_211, 1e-05);  convert_element_type_211 = None
    sqrt_105: "f32[104]" = torch.ops.aten.sqrt.default(add_264);  add_264 = None
    reciprocal_105: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
    mul_315: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
    unsqueeze_840: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_210, -1);  convert_element_type_210 = None
    unsqueeze_841: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
    unsqueeze_842: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
    unsqueeze_843: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
    sub_105: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_841);  convolution_105 = unsqueeze_841 = None
    mul_316: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
    unsqueeze_844: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
    unsqueeze_845: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
    mul_317: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
    unsqueeze_846: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
    unsqueeze_847: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
    add_265: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_102: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_265);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_102 = torch.ops.aten.split_with_sizes.default(relu_101, [104, 104, 104, 104], 1)
    getitem_411: "f32[8, 104, 14, 14]" = split_with_sizes_102[1];  split_with_sizes_102 = None
    add_266: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_102, getitem_411);  getitem_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_106: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_266, arg318_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_266 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_212: "f32[104]" = torch.ops.prims.convert_element_type.default(arg830_1, torch.float32);  arg830_1 = None
    convert_element_type_213: "f32[104]" = torch.ops.prims.convert_element_type.default(arg831_1, torch.float32);  arg831_1 = None
    add_267: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_213, 1e-05);  convert_element_type_213 = None
    sqrt_106: "f32[104]" = torch.ops.aten.sqrt.default(add_267);  add_267 = None
    reciprocal_106: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
    mul_318: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
    unsqueeze_848: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_212, -1);  convert_element_type_212 = None
    unsqueeze_849: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
    unsqueeze_850: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
    unsqueeze_851: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
    sub_106: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_849);  convolution_106 = unsqueeze_849 = None
    mul_319: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
    unsqueeze_852: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
    unsqueeze_853: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
    mul_320: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
    unsqueeze_854: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
    unsqueeze_855: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
    add_268: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_103: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_268);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_103 = torch.ops.aten.split_with_sizes.default(relu_101, [104, 104, 104, 104], 1)
    getitem_416: "f32[8, 104, 14, 14]" = split_with_sizes_103[2];  split_with_sizes_103 = None
    add_269: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_103, getitem_416);  getitem_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_107: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_269, arg321_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_269 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_214: "f32[104]" = torch.ops.prims.convert_element_type.default(arg833_1, torch.float32);  arg833_1 = None
    convert_element_type_215: "f32[104]" = torch.ops.prims.convert_element_type.default(arg834_1, torch.float32);  arg834_1 = None
    add_270: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_215, 1e-05);  convert_element_type_215 = None
    sqrt_107: "f32[104]" = torch.ops.aten.sqrt.default(add_270);  add_270 = None
    reciprocal_107: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
    mul_321: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
    unsqueeze_856: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_214, -1);  convert_element_type_214 = None
    unsqueeze_857: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
    unsqueeze_858: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
    unsqueeze_859: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
    sub_107: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_857);  convolution_107 = unsqueeze_857 = None
    mul_322: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
    unsqueeze_860: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
    unsqueeze_861: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
    mul_323: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
    unsqueeze_862: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
    unsqueeze_863: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
    add_271: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_104: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_271);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_104 = torch.ops.aten.split_with_sizes.default(relu_101, [104, 104, 104, 104], 1);  relu_101 = None
    getitem_421: "f32[8, 104, 14, 14]" = split_with_sizes_104[3];  split_with_sizes_104 = None
    cat_20: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_102, relu_103, relu_104, getitem_421], 1);  relu_102 = relu_103 = relu_104 = getitem_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_108: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_20, arg324_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_20 = arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_216: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg836_1, torch.float32);  arg836_1 = None
    convert_element_type_217: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg837_1, torch.float32);  arg837_1 = None
    add_272: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_217, 1e-05);  convert_element_type_217 = None
    sqrt_108: "f32[1024]" = torch.ops.aten.sqrt.default(add_272);  add_272 = None
    reciprocal_108: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
    mul_324: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
    unsqueeze_864: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_216, -1);  convert_element_type_216 = None
    unsqueeze_865: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
    unsqueeze_866: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
    unsqueeze_867: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
    sub_108: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_865);  convolution_108 = unsqueeze_865 = None
    mul_325: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
    unsqueeze_868: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
    unsqueeze_869: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
    mul_326: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
    unsqueeze_870: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
    unsqueeze_871: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
    add_273: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_274: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_273, relu_100);  add_273 = relu_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_105: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_274);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_109: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_105, arg327_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_218: "f32[416]" = torch.ops.prims.convert_element_type.default(arg839_1, torch.float32);  arg839_1 = None
    convert_element_type_219: "f32[416]" = torch.ops.prims.convert_element_type.default(arg840_1, torch.float32);  arg840_1 = None
    add_275: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_219, 1e-05);  convert_element_type_219 = None
    sqrt_109: "f32[416]" = torch.ops.aten.sqrt.default(add_275);  add_275 = None
    reciprocal_109: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
    mul_327: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
    unsqueeze_872: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_218, -1);  convert_element_type_218 = None
    unsqueeze_873: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
    unsqueeze_874: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
    unsqueeze_875: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
    sub_109: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_873);  convolution_109 = unsqueeze_873 = None
    mul_328: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
    unsqueeze_876: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
    unsqueeze_877: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
    mul_329: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
    unsqueeze_878: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
    unsqueeze_879: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
    add_276: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_106: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_276);  add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_106 = torch.ops.aten.split_with_sizes.default(relu_106, [104, 104, 104, 104], 1)
    getitem_426: "f32[8, 104, 14, 14]" = split_with_sizes_106[0];  split_with_sizes_106 = None
    convolution_110: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_426, arg330_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_426 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_220: "f32[104]" = torch.ops.prims.convert_element_type.default(arg842_1, torch.float32);  arg842_1 = None
    convert_element_type_221: "f32[104]" = torch.ops.prims.convert_element_type.default(arg843_1, torch.float32);  arg843_1 = None
    add_277: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_221, 1e-05);  convert_element_type_221 = None
    sqrt_110: "f32[104]" = torch.ops.aten.sqrt.default(add_277);  add_277 = None
    reciprocal_110: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
    mul_330: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
    unsqueeze_880: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_220, -1);  convert_element_type_220 = None
    unsqueeze_881: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
    unsqueeze_882: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
    unsqueeze_883: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
    sub_110: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_881);  convolution_110 = unsqueeze_881 = None
    mul_331: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
    unsqueeze_884: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
    unsqueeze_885: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
    mul_332: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
    unsqueeze_886: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
    unsqueeze_887: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
    add_278: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_107: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_278);  add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_107 = torch.ops.aten.split_with_sizes.default(relu_106, [104, 104, 104, 104], 1)
    getitem_431: "f32[8, 104, 14, 14]" = split_with_sizes_107[1];  split_with_sizes_107 = None
    add_279: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_107, getitem_431);  getitem_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_111: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_279, arg333_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_279 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_222: "f32[104]" = torch.ops.prims.convert_element_type.default(arg845_1, torch.float32);  arg845_1 = None
    convert_element_type_223: "f32[104]" = torch.ops.prims.convert_element_type.default(arg846_1, torch.float32);  arg846_1 = None
    add_280: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_223, 1e-05);  convert_element_type_223 = None
    sqrt_111: "f32[104]" = torch.ops.aten.sqrt.default(add_280);  add_280 = None
    reciprocal_111: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
    mul_333: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
    unsqueeze_888: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_222, -1);  convert_element_type_222 = None
    unsqueeze_889: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
    unsqueeze_890: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
    unsqueeze_891: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
    sub_111: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
    mul_334: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
    unsqueeze_892: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
    unsqueeze_893: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
    mul_335: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
    unsqueeze_894: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
    unsqueeze_895: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
    add_281: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_108: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_281);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_108 = torch.ops.aten.split_with_sizes.default(relu_106, [104, 104, 104, 104], 1)
    getitem_436: "f32[8, 104, 14, 14]" = split_with_sizes_108[2];  split_with_sizes_108 = None
    add_282: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_108, getitem_436);  getitem_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_112: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_282, arg336_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_282 = arg336_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_224: "f32[104]" = torch.ops.prims.convert_element_type.default(arg848_1, torch.float32);  arg848_1 = None
    convert_element_type_225: "f32[104]" = torch.ops.prims.convert_element_type.default(arg849_1, torch.float32);  arg849_1 = None
    add_283: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_225, 1e-05);  convert_element_type_225 = None
    sqrt_112: "f32[104]" = torch.ops.aten.sqrt.default(add_283);  add_283 = None
    reciprocal_112: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
    mul_336: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
    unsqueeze_896: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_224, -1);  convert_element_type_224 = None
    unsqueeze_897: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
    unsqueeze_898: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
    unsqueeze_899: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
    sub_112: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_897);  convolution_112 = unsqueeze_897 = None
    mul_337: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
    unsqueeze_900: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
    unsqueeze_901: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
    mul_338: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
    unsqueeze_902: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
    unsqueeze_903: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
    add_284: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_109: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_284);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_109 = torch.ops.aten.split_with_sizes.default(relu_106, [104, 104, 104, 104], 1);  relu_106 = None
    getitem_441: "f32[8, 104, 14, 14]" = split_with_sizes_109[3];  split_with_sizes_109 = None
    cat_21: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_107, relu_108, relu_109, getitem_441], 1);  relu_107 = relu_108 = relu_109 = getitem_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_113: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_21, arg339_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_21 = arg339_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_226: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg851_1, torch.float32);  arg851_1 = None
    convert_element_type_227: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg852_1, torch.float32);  arg852_1 = None
    add_285: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_227, 1e-05);  convert_element_type_227 = None
    sqrt_113: "f32[1024]" = torch.ops.aten.sqrt.default(add_285);  add_285 = None
    reciprocal_113: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
    mul_339: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
    unsqueeze_904: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_226, -1);  convert_element_type_226 = None
    unsqueeze_905: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
    unsqueeze_906: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
    unsqueeze_907: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
    sub_113: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_905);  convolution_113 = unsqueeze_905 = None
    mul_340: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
    unsqueeze_908: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
    unsqueeze_909: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
    mul_341: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
    unsqueeze_910: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
    unsqueeze_911: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
    add_286: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_287: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_286, relu_105);  add_286 = relu_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_110: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_287);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_114: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_110, arg342_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg342_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_228: "f32[416]" = torch.ops.prims.convert_element_type.default(arg854_1, torch.float32);  arg854_1 = None
    convert_element_type_229: "f32[416]" = torch.ops.prims.convert_element_type.default(arg855_1, torch.float32);  arg855_1 = None
    add_288: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_229, 1e-05);  convert_element_type_229 = None
    sqrt_114: "f32[416]" = torch.ops.aten.sqrt.default(add_288);  add_288 = None
    reciprocal_114: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
    mul_342: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
    unsqueeze_912: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_228, -1);  convert_element_type_228 = None
    unsqueeze_913: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
    unsqueeze_914: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
    unsqueeze_915: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
    sub_114: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_913);  convolution_114 = unsqueeze_913 = None
    mul_343: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
    unsqueeze_916: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
    unsqueeze_917: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
    mul_344: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
    unsqueeze_918: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
    unsqueeze_919: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
    add_289: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_111: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_289);  add_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_111 = torch.ops.aten.split_with_sizes.default(relu_111, [104, 104, 104, 104], 1)
    getitem_446: "f32[8, 104, 14, 14]" = split_with_sizes_111[0];  split_with_sizes_111 = None
    convolution_115: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_446, arg345_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_446 = arg345_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_230: "f32[104]" = torch.ops.prims.convert_element_type.default(arg857_1, torch.float32);  arg857_1 = None
    convert_element_type_231: "f32[104]" = torch.ops.prims.convert_element_type.default(arg858_1, torch.float32);  arg858_1 = None
    add_290: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_231, 1e-05);  convert_element_type_231 = None
    sqrt_115: "f32[104]" = torch.ops.aten.sqrt.default(add_290);  add_290 = None
    reciprocal_115: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
    mul_345: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
    unsqueeze_920: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_230, -1);  convert_element_type_230 = None
    unsqueeze_921: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
    unsqueeze_922: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
    unsqueeze_923: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
    sub_115: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_921);  convolution_115 = unsqueeze_921 = None
    mul_346: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
    unsqueeze_924: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
    unsqueeze_925: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
    mul_347: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
    unsqueeze_926: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
    unsqueeze_927: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
    add_291: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_112: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_291);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_112 = torch.ops.aten.split_with_sizes.default(relu_111, [104, 104, 104, 104], 1)
    getitem_451: "f32[8, 104, 14, 14]" = split_with_sizes_112[1];  split_with_sizes_112 = None
    add_292: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_112, getitem_451);  getitem_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_116: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_292, arg348_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_292 = arg348_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_232: "f32[104]" = torch.ops.prims.convert_element_type.default(arg860_1, torch.float32);  arg860_1 = None
    convert_element_type_233: "f32[104]" = torch.ops.prims.convert_element_type.default(arg861_1, torch.float32);  arg861_1 = None
    add_293: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_233, 1e-05);  convert_element_type_233 = None
    sqrt_116: "f32[104]" = torch.ops.aten.sqrt.default(add_293);  add_293 = None
    reciprocal_116: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
    mul_348: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
    unsqueeze_928: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_232, -1);  convert_element_type_232 = None
    unsqueeze_929: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
    unsqueeze_930: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
    unsqueeze_931: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
    sub_116: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_929);  convolution_116 = unsqueeze_929 = None
    mul_349: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
    unsqueeze_932: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
    unsqueeze_933: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
    mul_350: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
    unsqueeze_934: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
    unsqueeze_935: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
    add_294: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_113: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_294);  add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_113 = torch.ops.aten.split_with_sizes.default(relu_111, [104, 104, 104, 104], 1)
    getitem_456: "f32[8, 104, 14, 14]" = split_with_sizes_113[2];  split_with_sizes_113 = None
    add_295: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_113, getitem_456);  getitem_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_117: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_295, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_295 = arg351_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_234: "f32[104]" = torch.ops.prims.convert_element_type.default(arg863_1, torch.float32);  arg863_1 = None
    convert_element_type_235: "f32[104]" = torch.ops.prims.convert_element_type.default(arg864_1, torch.float32);  arg864_1 = None
    add_296: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_235, 1e-05);  convert_element_type_235 = None
    sqrt_117: "f32[104]" = torch.ops.aten.sqrt.default(add_296);  add_296 = None
    reciprocal_117: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
    mul_351: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
    unsqueeze_936: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_234, -1);  convert_element_type_234 = None
    unsqueeze_937: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
    unsqueeze_938: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
    unsqueeze_939: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
    sub_117: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_937);  convolution_117 = unsqueeze_937 = None
    mul_352: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
    unsqueeze_940: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
    unsqueeze_941: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
    mul_353: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
    unsqueeze_942: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg353_1, -1);  arg353_1 = None
    unsqueeze_943: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
    add_297: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_114: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_297);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_114 = torch.ops.aten.split_with_sizes.default(relu_111, [104, 104, 104, 104], 1);  relu_111 = None
    getitem_461: "f32[8, 104, 14, 14]" = split_with_sizes_114[3];  split_with_sizes_114 = None
    cat_22: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_112, relu_113, relu_114, getitem_461], 1);  relu_112 = relu_113 = relu_114 = getitem_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_118: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_22, arg354_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_22 = arg354_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_236: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg866_1, torch.float32);  arg866_1 = None
    convert_element_type_237: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg867_1, torch.float32);  arg867_1 = None
    add_298: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_237, 1e-05);  convert_element_type_237 = None
    sqrt_118: "f32[1024]" = torch.ops.aten.sqrt.default(add_298);  add_298 = None
    reciprocal_118: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
    mul_354: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
    unsqueeze_944: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_236, -1);  convert_element_type_236 = None
    unsqueeze_945: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
    unsqueeze_946: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
    unsqueeze_947: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
    sub_118: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_945);  convolution_118 = unsqueeze_945 = None
    mul_355: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
    unsqueeze_948: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
    unsqueeze_949: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
    mul_356: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
    unsqueeze_950: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg356_1, -1);  arg356_1 = None
    unsqueeze_951: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
    add_299: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_299, relu_110);  add_299 = relu_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_115: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_300);  add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_119: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_115, arg357_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg357_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_238: "f32[416]" = torch.ops.prims.convert_element_type.default(arg869_1, torch.float32);  arg869_1 = None
    convert_element_type_239: "f32[416]" = torch.ops.prims.convert_element_type.default(arg870_1, torch.float32);  arg870_1 = None
    add_301: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_239, 1e-05);  convert_element_type_239 = None
    sqrt_119: "f32[416]" = torch.ops.aten.sqrt.default(add_301);  add_301 = None
    reciprocal_119: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
    mul_357: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
    unsqueeze_952: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_238, -1);  convert_element_type_238 = None
    unsqueeze_953: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
    unsqueeze_954: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
    unsqueeze_955: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
    sub_119: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_953);  convolution_119 = unsqueeze_953 = None
    mul_358: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
    unsqueeze_956: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
    unsqueeze_957: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
    mul_359: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
    unsqueeze_958: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
    unsqueeze_959: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
    add_302: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_116: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_302);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_116 = torch.ops.aten.split_with_sizes.default(relu_116, [104, 104, 104, 104], 1)
    getitem_466: "f32[8, 104, 14, 14]" = split_with_sizes_116[0];  split_with_sizes_116 = None
    convolution_120: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_466, arg360_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_466 = arg360_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_240: "f32[104]" = torch.ops.prims.convert_element_type.default(arg872_1, torch.float32);  arg872_1 = None
    convert_element_type_241: "f32[104]" = torch.ops.prims.convert_element_type.default(arg873_1, torch.float32);  arg873_1 = None
    add_303: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_241, 1e-05);  convert_element_type_241 = None
    sqrt_120: "f32[104]" = torch.ops.aten.sqrt.default(add_303);  add_303 = None
    reciprocal_120: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
    mul_360: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
    unsqueeze_960: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_240, -1);  convert_element_type_240 = None
    unsqueeze_961: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
    unsqueeze_962: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
    unsqueeze_963: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
    sub_120: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_961);  convolution_120 = unsqueeze_961 = None
    mul_361: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
    unsqueeze_964: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
    unsqueeze_965: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
    mul_362: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
    unsqueeze_966: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
    unsqueeze_967: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
    add_304: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_117: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_304);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_117 = torch.ops.aten.split_with_sizes.default(relu_116, [104, 104, 104, 104], 1)
    getitem_471: "f32[8, 104, 14, 14]" = split_with_sizes_117[1];  split_with_sizes_117 = None
    add_305: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_117, getitem_471);  getitem_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_121: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_305, arg363_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_305 = arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_242: "f32[104]" = torch.ops.prims.convert_element_type.default(arg875_1, torch.float32);  arg875_1 = None
    convert_element_type_243: "f32[104]" = torch.ops.prims.convert_element_type.default(arg876_1, torch.float32);  arg876_1 = None
    add_306: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_243, 1e-05);  convert_element_type_243 = None
    sqrt_121: "f32[104]" = torch.ops.aten.sqrt.default(add_306);  add_306 = None
    reciprocal_121: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
    mul_363: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
    unsqueeze_968: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_242, -1);  convert_element_type_242 = None
    unsqueeze_969: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
    unsqueeze_970: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
    unsqueeze_971: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
    sub_121: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_969);  convolution_121 = unsqueeze_969 = None
    mul_364: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
    unsqueeze_972: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
    unsqueeze_973: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
    mul_365: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
    unsqueeze_974: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
    unsqueeze_975: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
    add_307: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_118: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_307);  add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_118 = torch.ops.aten.split_with_sizes.default(relu_116, [104, 104, 104, 104], 1)
    getitem_476: "f32[8, 104, 14, 14]" = split_with_sizes_118[2];  split_with_sizes_118 = None
    add_308: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_118, getitem_476);  getitem_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_122: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_308, arg366_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_308 = arg366_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_244: "f32[104]" = torch.ops.prims.convert_element_type.default(arg878_1, torch.float32);  arg878_1 = None
    convert_element_type_245: "f32[104]" = torch.ops.prims.convert_element_type.default(arg879_1, torch.float32);  arg879_1 = None
    add_309: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_245, 1e-05);  convert_element_type_245 = None
    sqrt_122: "f32[104]" = torch.ops.aten.sqrt.default(add_309);  add_309 = None
    reciprocal_122: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
    mul_366: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
    unsqueeze_976: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_244, -1);  convert_element_type_244 = None
    unsqueeze_977: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
    unsqueeze_978: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
    unsqueeze_979: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
    sub_122: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_977);  convolution_122 = unsqueeze_977 = None
    mul_367: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
    unsqueeze_980: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
    unsqueeze_981: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
    mul_368: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_981);  mul_367 = unsqueeze_981 = None
    unsqueeze_982: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
    unsqueeze_983: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
    add_310: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_368, unsqueeze_983);  mul_368 = unsqueeze_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_119: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_310);  add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_119 = torch.ops.aten.split_with_sizes.default(relu_116, [104, 104, 104, 104], 1);  relu_116 = None
    getitem_481: "f32[8, 104, 14, 14]" = split_with_sizes_119[3];  split_with_sizes_119 = None
    cat_23: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_117, relu_118, relu_119, getitem_481], 1);  relu_117 = relu_118 = relu_119 = getitem_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_123: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_23, arg369_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_23 = arg369_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_246: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg881_1, torch.float32);  arg881_1 = None
    convert_element_type_247: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg882_1, torch.float32);  arg882_1 = None
    add_311: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_247, 1e-05);  convert_element_type_247 = None
    sqrt_123: "f32[1024]" = torch.ops.aten.sqrt.default(add_311);  add_311 = None
    reciprocal_123: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
    mul_369: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
    unsqueeze_984: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_246, -1);  convert_element_type_246 = None
    unsqueeze_985: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
    unsqueeze_986: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
    unsqueeze_987: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
    sub_123: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_985);  convolution_123 = unsqueeze_985 = None
    mul_370: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
    unsqueeze_988: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
    unsqueeze_989: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
    mul_371: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_989);  mul_370 = unsqueeze_989 = None
    unsqueeze_990: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
    unsqueeze_991: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
    add_312: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_991);  mul_371 = unsqueeze_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_313: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_312, relu_115);  add_312 = relu_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_120: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_313);  add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_124: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_120, arg372_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg372_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_248: "f32[416]" = torch.ops.prims.convert_element_type.default(arg884_1, torch.float32);  arg884_1 = None
    convert_element_type_249: "f32[416]" = torch.ops.prims.convert_element_type.default(arg885_1, torch.float32);  arg885_1 = None
    add_314: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_249, 1e-05);  convert_element_type_249 = None
    sqrt_124: "f32[416]" = torch.ops.aten.sqrt.default(add_314);  add_314 = None
    reciprocal_124: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
    mul_372: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
    unsqueeze_992: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_248, -1);  convert_element_type_248 = None
    unsqueeze_993: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
    unsqueeze_994: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
    unsqueeze_995: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
    sub_124: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_993);  convolution_124 = unsqueeze_993 = None
    mul_373: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
    unsqueeze_996: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
    unsqueeze_997: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
    mul_374: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_997);  mul_373 = unsqueeze_997 = None
    unsqueeze_998: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
    unsqueeze_999: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
    add_315: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_999);  mul_374 = unsqueeze_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_121: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_315);  add_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_121 = torch.ops.aten.split_with_sizes.default(relu_121, [104, 104, 104, 104], 1)
    getitem_486: "f32[8, 104, 14, 14]" = split_with_sizes_121[0];  split_with_sizes_121 = None
    convolution_125: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_486, arg375_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_486 = arg375_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_250: "f32[104]" = torch.ops.prims.convert_element_type.default(arg887_1, torch.float32);  arg887_1 = None
    convert_element_type_251: "f32[104]" = torch.ops.prims.convert_element_type.default(arg888_1, torch.float32);  arg888_1 = None
    add_316: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_251, 1e-05);  convert_element_type_251 = None
    sqrt_125: "f32[104]" = torch.ops.aten.sqrt.default(add_316);  add_316 = None
    reciprocal_125: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
    mul_375: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
    unsqueeze_1000: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_250, -1);  convert_element_type_250 = None
    unsqueeze_1001: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
    unsqueeze_1003: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
    sub_125: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1001);  convolution_125 = unsqueeze_1001 = None
    mul_376: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
    unsqueeze_1004: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
    unsqueeze_1005: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
    mul_377: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_1005);  mul_376 = unsqueeze_1005 = None
    unsqueeze_1006: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
    unsqueeze_1007: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
    add_317: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_1007);  mul_377 = unsqueeze_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_122: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_317);  add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_122 = torch.ops.aten.split_with_sizes.default(relu_121, [104, 104, 104, 104], 1)
    getitem_491: "f32[8, 104, 14, 14]" = split_with_sizes_122[1];  split_with_sizes_122 = None
    add_318: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_122, getitem_491);  getitem_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_126: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_318, arg378_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_318 = arg378_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_252: "f32[104]" = torch.ops.prims.convert_element_type.default(arg890_1, torch.float32);  arg890_1 = None
    convert_element_type_253: "f32[104]" = torch.ops.prims.convert_element_type.default(arg891_1, torch.float32);  arg891_1 = None
    add_319: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_253, 1e-05);  convert_element_type_253 = None
    sqrt_126: "f32[104]" = torch.ops.aten.sqrt.default(add_319);  add_319 = None
    reciprocal_126: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
    mul_378: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
    unsqueeze_1008: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_252, -1);  convert_element_type_252 = None
    unsqueeze_1009: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
    unsqueeze_1010: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
    unsqueeze_1011: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
    sub_126: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1009);  convolution_126 = unsqueeze_1009 = None
    mul_379: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
    unsqueeze_1012: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
    unsqueeze_1013: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
    mul_380: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_1013);  mul_379 = unsqueeze_1013 = None
    unsqueeze_1014: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
    unsqueeze_1015: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
    add_320: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_1015);  mul_380 = unsqueeze_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_123: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_320);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_123 = torch.ops.aten.split_with_sizes.default(relu_121, [104, 104, 104, 104], 1)
    getitem_496: "f32[8, 104, 14, 14]" = split_with_sizes_123[2];  split_with_sizes_123 = None
    add_321: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_123, getitem_496);  getitem_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_127: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_321, arg381_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_321 = arg381_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_254: "f32[104]" = torch.ops.prims.convert_element_type.default(arg893_1, torch.float32);  arg893_1 = None
    convert_element_type_255: "f32[104]" = torch.ops.prims.convert_element_type.default(arg894_1, torch.float32);  arg894_1 = None
    add_322: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_255, 1e-05);  convert_element_type_255 = None
    sqrt_127: "f32[104]" = torch.ops.aten.sqrt.default(add_322);  add_322 = None
    reciprocal_127: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
    mul_381: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
    unsqueeze_1016: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_254, -1);  convert_element_type_254 = None
    unsqueeze_1017: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
    unsqueeze_1019: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
    sub_127: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1017);  convolution_127 = unsqueeze_1017 = None
    mul_382: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
    unsqueeze_1020: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
    unsqueeze_1021: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
    mul_383: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_1021);  mul_382 = unsqueeze_1021 = None
    unsqueeze_1022: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg383_1, -1);  arg383_1 = None
    unsqueeze_1023: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
    add_323: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_1023);  mul_383 = unsqueeze_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_124: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_323);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_124 = torch.ops.aten.split_with_sizes.default(relu_121, [104, 104, 104, 104], 1);  relu_121 = None
    getitem_501: "f32[8, 104, 14, 14]" = split_with_sizes_124[3];  split_with_sizes_124 = None
    cat_24: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_122, relu_123, relu_124, getitem_501], 1);  relu_122 = relu_123 = relu_124 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_24, arg384_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_24 = arg384_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_256: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg896_1, torch.float32);  arg896_1 = None
    convert_element_type_257: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg897_1, torch.float32);  arg897_1 = None
    add_324: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_257, 1e-05);  convert_element_type_257 = None
    sqrt_128: "f32[1024]" = torch.ops.aten.sqrt.default(add_324);  add_324 = None
    reciprocal_128: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
    mul_384: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
    unsqueeze_1024: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_256, -1);  convert_element_type_256 = None
    unsqueeze_1025: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
    unsqueeze_1027: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
    sub_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_1025);  convolution_128 = unsqueeze_1025 = None
    mul_385: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
    unsqueeze_1028: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
    unsqueeze_1029: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
    mul_386: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1029);  mul_385 = unsqueeze_1029 = None
    unsqueeze_1030: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
    unsqueeze_1031: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
    add_325: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1031);  mul_386 = unsqueeze_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_326: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_325, relu_120);  add_325 = relu_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_125: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_326);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_129: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_125, arg387_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg387_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_258: "f32[416]" = torch.ops.prims.convert_element_type.default(arg899_1, torch.float32);  arg899_1 = None
    convert_element_type_259: "f32[416]" = torch.ops.prims.convert_element_type.default(arg900_1, torch.float32);  arg900_1 = None
    add_327: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_259, 1e-05);  convert_element_type_259 = None
    sqrt_129: "f32[416]" = torch.ops.aten.sqrt.default(add_327);  add_327 = None
    reciprocal_129: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
    mul_387: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
    unsqueeze_1032: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_258, -1);  convert_element_type_258 = None
    unsqueeze_1033: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
    unsqueeze_1034: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
    unsqueeze_1035: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
    sub_129: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1033);  convolution_129 = unsqueeze_1033 = None
    mul_388: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
    unsqueeze_1036: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
    unsqueeze_1037: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
    mul_389: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1037);  mul_388 = unsqueeze_1037 = None
    unsqueeze_1038: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
    unsqueeze_1039: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
    add_328: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1039);  mul_389 = unsqueeze_1039 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_126: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_328);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_126 = torch.ops.aten.split_with_sizes.default(relu_126, [104, 104, 104, 104], 1)
    getitem_506: "f32[8, 104, 14, 14]" = split_with_sizes_126[0];  split_with_sizes_126 = None
    convolution_130: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_506, arg390_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_506 = arg390_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_260: "f32[104]" = torch.ops.prims.convert_element_type.default(arg902_1, torch.float32);  arg902_1 = None
    convert_element_type_261: "f32[104]" = torch.ops.prims.convert_element_type.default(arg903_1, torch.float32);  arg903_1 = None
    add_329: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_261, 1e-05);  convert_element_type_261 = None
    sqrt_130: "f32[104]" = torch.ops.aten.sqrt.default(add_329);  add_329 = None
    reciprocal_130: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
    mul_390: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
    unsqueeze_1040: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_260, -1);  convert_element_type_260 = None
    unsqueeze_1041: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
    unsqueeze_1042: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
    unsqueeze_1043: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
    sub_130: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1041);  convolution_130 = unsqueeze_1041 = None
    mul_391: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
    unsqueeze_1044: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
    unsqueeze_1045: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
    mul_392: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1045);  mul_391 = unsqueeze_1045 = None
    unsqueeze_1046: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
    unsqueeze_1047: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
    add_330: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1047);  mul_392 = unsqueeze_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_127: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_330);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_127 = torch.ops.aten.split_with_sizes.default(relu_126, [104, 104, 104, 104], 1)
    getitem_511: "f32[8, 104, 14, 14]" = split_with_sizes_127[1];  split_with_sizes_127 = None
    add_331: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_127, getitem_511);  getitem_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_131: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_331, arg393_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_331 = arg393_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_262: "f32[104]" = torch.ops.prims.convert_element_type.default(arg905_1, torch.float32);  arg905_1 = None
    convert_element_type_263: "f32[104]" = torch.ops.prims.convert_element_type.default(arg906_1, torch.float32);  arg906_1 = None
    add_332: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_263, 1e-05);  convert_element_type_263 = None
    sqrt_131: "f32[104]" = torch.ops.aten.sqrt.default(add_332);  add_332 = None
    reciprocal_131: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
    mul_393: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
    unsqueeze_1048: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_262, -1);  convert_element_type_262 = None
    unsqueeze_1049: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
    unsqueeze_1051: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
    sub_131: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_1049);  convolution_131 = unsqueeze_1049 = None
    mul_394: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
    unsqueeze_1052: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
    unsqueeze_1053: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
    mul_395: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_1053);  mul_394 = unsqueeze_1053 = None
    unsqueeze_1054: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
    unsqueeze_1055: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
    add_333: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_1055);  mul_395 = unsqueeze_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_128: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_333);  add_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_128 = torch.ops.aten.split_with_sizes.default(relu_126, [104, 104, 104, 104], 1)
    getitem_516: "f32[8, 104, 14, 14]" = split_with_sizes_128[2];  split_with_sizes_128 = None
    add_334: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_128, getitem_516);  getitem_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_132: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_334, arg396_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_334 = arg396_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_264: "f32[104]" = torch.ops.prims.convert_element_type.default(arg908_1, torch.float32);  arg908_1 = None
    convert_element_type_265: "f32[104]" = torch.ops.prims.convert_element_type.default(arg909_1, torch.float32);  arg909_1 = None
    add_335: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_265, 1e-05);  convert_element_type_265 = None
    sqrt_132: "f32[104]" = torch.ops.aten.sqrt.default(add_335);  add_335 = None
    reciprocal_132: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
    mul_396: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
    unsqueeze_1056: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_264, -1);  convert_element_type_264 = None
    unsqueeze_1057: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
    unsqueeze_1058: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
    unsqueeze_1059: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
    sub_132: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1057);  convolution_132 = unsqueeze_1057 = None
    mul_397: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
    unsqueeze_1060: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
    unsqueeze_1061: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
    mul_398: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_1061);  mul_397 = unsqueeze_1061 = None
    unsqueeze_1062: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
    unsqueeze_1063: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
    add_336: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_1063);  mul_398 = unsqueeze_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_129: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_336);  add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_129 = torch.ops.aten.split_with_sizes.default(relu_126, [104, 104, 104, 104], 1);  relu_126 = None
    getitem_521: "f32[8, 104, 14, 14]" = split_with_sizes_129[3];  split_with_sizes_129 = None
    cat_25: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_127, relu_128, relu_129, getitem_521], 1);  relu_127 = relu_128 = relu_129 = getitem_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_133: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_25, arg399_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_25 = arg399_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_266: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg911_1, torch.float32);  arg911_1 = None
    convert_element_type_267: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg912_1, torch.float32);  arg912_1 = None
    add_337: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_267, 1e-05);  convert_element_type_267 = None
    sqrt_133: "f32[1024]" = torch.ops.aten.sqrt.default(add_337);  add_337 = None
    reciprocal_133: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
    mul_399: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
    unsqueeze_1064: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_266, -1);  convert_element_type_266 = None
    unsqueeze_1065: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
    unsqueeze_1066: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
    unsqueeze_1067: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
    sub_133: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1065);  convolution_133 = unsqueeze_1065 = None
    mul_400: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
    unsqueeze_1068: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
    unsqueeze_1069: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
    mul_401: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_1069);  mul_400 = unsqueeze_1069 = None
    unsqueeze_1070: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
    unsqueeze_1071: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
    add_338: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_1071);  mul_401 = unsqueeze_1071 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_339: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_338, relu_125);  add_338 = relu_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_130: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_339);  add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_134: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_130, arg402_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg402_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_268: "f32[416]" = torch.ops.prims.convert_element_type.default(arg914_1, torch.float32);  arg914_1 = None
    convert_element_type_269: "f32[416]" = torch.ops.prims.convert_element_type.default(arg915_1, torch.float32);  arg915_1 = None
    add_340: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_269, 1e-05);  convert_element_type_269 = None
    sqrt_134: "f32[416]" = torch.ops.aten.sqrt.default(add_340);  add_340 = None
    reciprocal_134: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
    mul_402: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
    unsqueeze_1072: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_268, -1);  convert_element_type_268 = None
    unsqueeze_1073: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
    unsqueeze_1074: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_402, -1);  mul_402 = None
    unsqueeze_1075: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
    sub_134: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_1073);  convolution_134 = unsqueeze_1073 = None
    mul_403: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
    unsqueeze_1076: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
    unsqueeze_1077: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
    mul_404: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_1077);  mul_403 = unsqueeze_1077 = None
    unsqueeze_1078: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
    unsqueeze_1079: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
    add_341: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_404, unsqueeze_1079);  mul_404 = unsqueeze_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_131: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_341);  add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_131 = torch.ops.aten.split_with_sizes.default(relu_131, [104, 104, 104, 104], 1)
    getitem_526: "f32[8, 104, 14, 14]" = split_with_sizes_131[0];  split_with_sizes_131 = None
    convolution_135: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_526, arg405_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_526 = arg405_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_270: "f32[104]" = torch.ops.prims.convert_element_type.default(arg917_1, torch.float32);  arg917_1 = None
    convert_element_type_271: "f32[104]" = torch.ops.prims.convert_element_type.default(arg918_1, torch.float32);  arg918_1 = None
    add_342: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_271, 1e-05);  convert_element_type_271 = None
    sqrt_135: "f32[104]" = torch.ops.aten.sqrt.default(add_342);  add_342 = None
    reciprocal_135: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
    mul_405: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
    unsqueeze_1080: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_270, -1);  convert_element_type_270 = None
    unsqueeze_1081: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
    unsqueeze_1082: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
    unsqueeze_1083: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
    sub_135: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_1081);  convolution_135 = unsqueeze_1081 = None
    mul_406: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
    unsqueeze_1084: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
    unsqueeze_1085: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
    mul_407: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_1085);  mul_406 = unsqueeze_1085 = None
    unsqueeze_1086: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
    unsqueeze_1087: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
    add_343: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_407, unsqueeze_1087);  mul_407 = unsqueeze_1087 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_132: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_343);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_132 = torch.ops.aten.split_with_sizes.default(relu_131, [104, 104, 104, 104], 1)
    getitem_531: "f32[8, 104, 14, 14]" = split_with_sizes_132[1];  split_with_sizes_132 = None
    add_344: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_132, getitem_531);  getitem_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_136: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_344, arg408_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_344 = arg408_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_272: "f32[104]" = torch.ops.prims.convert_element_type.default(arg920_1, torch.float32);  arg920_1 = None
    convert_element_type_273: "f32[104]" = torch.ops.prims.convert_element_type.default(arg921_1, torch.float32);  arg921_1 = None
    add_345: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_273, 1e-05);  convert_element_type_273 = None
    sqrt_136: "f32[104]" = torch.ops.aten.sqrt.default(add_345);  add_345 = None
    reciprocal_136: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
    mul_408: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
    unsqueeze_1088: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_272, -1);  convert_element_type_272 = None
    unsqueeze_1089: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
    unsqueeze_1090: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
    unsqueeze_1091: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
    sub_136: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_1089);  convolution_136 = unsqueeze_1089 = None
    mul_409: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
    unsqueeze_1092: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
    unsqueeze_1093: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
    mul_410: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_1093);  mul_409 = unsqueeze_1093 = None
    unsqueeze_1094: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
    unsqueeze_1095: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
    add_346: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_1095);  mul_410 = unsqueeze_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_133: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_346);  add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_133 = torch.ops.aten.split_with_sizes.default(relu_131, [104, 104, 104, 104], 1)
    getitem_536: "f32[8, 104, 14, 14]" = split_with_sizes_133[2];  split_with_sizes_133 = None
    add_347: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_133, getitem_536);  getitem_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_137: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_347, arg411_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_347 = arg411_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_274: "f32[104]" = torch.ops.prims.convert_element_type.default(arg923_1, torch.float32);  arg923_1 = None
    convert_element_type_275: "f32[104]" = torch.ops.prims.convert_element_type.default(arg924_1, torch.float32);  arg924_1 = None
    add_348: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_275, 1e-05);  convert_element_type_275 = None
    sqrt_137: "f32[104]" = torch.ops.aten.sqrt.default(add_348);  add_348 = None
    reciprocal_137: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
    mul_411: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
    unsqueeze_1096: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_274, -1);  convert_element_type_274 = None
    unsqueeze_1097: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
    unsqueeze_1098: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
    unsqueeze_1099: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
    sub_137: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_1097);  convolution_137 = unsqueeze_1097 = None
    mul_412: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
    unsqueeze_1100: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
    unsqueeze_1101: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
    mul_413: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_1101);  mul_412 = unsqueeze_1101 = None
    unsqueeze_1102: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
    unsqueeze_1103: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
    add_349: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_1103);  mul_413 = unsqueeze_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_134: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_349);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_134 = torch.ops.aten.split_with_sizes.default(relu_131, [104, 104, 104, 104], 1);  relu_131 = None
    getitem_541: "f32[8, 104, 14, 14]" = split_with_sizes_134[3];  split_with_sizes_134 = None
    cat_26: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_132, relu_133, relu_134, getitem_541], 1);  relu_132 = relu_133 = relu_134 = getitem_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_138: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_26, arg414_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_26 = arg414_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_276: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg926_1, torch.float32);  arg926_1 = None
    convert_element_type_277: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg927_1, torch.float32);  arg927_1 = None
    add_350: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_277, 1e-05);  convert_element_type_277 = None
    sqrt_138: "f32[1024]" = torch.ops.aten.sqrt.default(add_350);  add_350 = None
    reciprocal_138: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
    mul_414: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
    unsqueeze_1104: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_276, -1);  convert_element_type_276 = None
    unsqueeze_1105: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
    unsqueeze_1106: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
    unsqueeze_1107: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
    sub_138: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_1105);  convolution_138 = unsqueeze_1105 = None
    mul_415: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
    unsqueeze_1108: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
    unsqueeze_1109: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
    mul_416: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_1109);  mul_415 = unsqueeze_1109 = None
    unsqueeze_1110: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
    unsqueeze_1111: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
    add_351: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_1111);  mul_416 = unsqueeze_1111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_352: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_351, relu_130);  add_351 = relu_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_135: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_352);  add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_139: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_135, arg417_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg417_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_278: "f32[416]" = torch.ops.prims.convert_element_type.default(arg929_1, torch.float32);  arg929_1 = None
    convert_element_type_279: "f32[416]" = torch.ops.prims.convert_element_type.default(arg930_1, torch.float32);  arg930_1 = None
    add_353: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_279, 1e-05);  convert_element_type_279 = None
    sqrt_139: "f32[416]" = torch.ops.aten.sqrt.default(add_353);  add_353 = None
    reciprocal_139: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
    mul_417: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
    unsqueeze_1112: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_278, -1);  convert_element_type_278 = None
    unsqueeze_1113: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
    unsqueeze_1114: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
    unsqueeze_1115: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
    sub_139: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_1113);  convolution_139 = unsqueeze_1113 = None
    mul_418: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
    unsqueeze_1116: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
    unsqueeze_1117: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
    mul_419: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_1117);  mul_418 = unsqueeze_1117 = None
    unsqueeze_1118: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
    unsqueeze_1119: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
    add_354: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_1119);  mul_419 = unsqueeze_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_136: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_354);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_136 = torch.ops.aten.split_with_sizes.default(relu_136, [104, 104, 104, 104], 1)
    getitem_546: "f32[8, 104, 14, 14]" = split_with_sizes_136[0];  split_with_sizes_136 = None
    convolution_140: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_546, arg420_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_546 = arg420_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_280: "f32[104]" = torch.ops.prims.convert_element_type.default(arg932_1, torch.float32);  arg932_1 = None
    convert_element_type_281: "f32[104]" = torch.ops.prims.convert_element_type.default(arg933_1, torch.float32);  arg933_1 = None
    add_355: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_281, 1e-05);  convert_element_type_281 = None
    sqrt_140: "f32[104]" = torch.ops.aten.sqrt.default(add_355);  add_355 = None
    reciprocal_140: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
    mul_420: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
    unsqueeze_1120: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_280, -1);  convert_element_type_280 = None
    unsqueeze_1121: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
    unsqueeze_1122: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
    unsqueeze_1123: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
    sub_140: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_1121);  convolution_140 = unsqueeze_1121 = None
    mul_421: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
    unsqueeze_1124: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
    unsqueeze_1125: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
    mul_422: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_1125);  mul_421 = unsqueeze_1125 = None
    unsqueeze_1126: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
    unsqueeze_1127: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
    add_356: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_1127);  mul_422 = unsqueeze_1127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_137: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_356);  add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_137 = torch.ops.aten.split_with_sizes.default(relu_136, [104, 104, 104, 104], 1)
    getitem_551: "f32[8, 104, 14, 14]" = split_with_sizes_137[1];  split_with_sizes_137 = None
    add_357: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_137, getitem_551);  getitem_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_141: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_357, arg423_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_357 = arg423_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_282: "f32[104]" = torch.ops.prims.convert_element_type.default(arg935_1, torch.float32);  arg935_1 = None
    convert_element_type_283: "f32[104]" = torch.ops.prims.convert_element_type.default(arg936_1, torch.float32);  arg936_1 = None
    add_358: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_283, 1e-05);  convert_element_type_283 = None
    sqrt_141: "f32[104]" = torch.ops.aten.sqrt.default(add_358);  add_358 = None
    reciprocal_141: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
    mul_423: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
    unsqueeze_1128: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_282, -1);  convert_element_type_282 = None
    unsqueeze_1129: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
    unsqueeze_1130: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_423, -1);  mul_423 = None
    unsqueeze_1131: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
    sub_141: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_1129);  convolution_141 = unsqueeze_1129 = None
    mul_424: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
    unsqueeze_1132: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
    unsqueeze_1133: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
    mul_425: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_1133);  mul_424 = unsqueeze_1133 = None
    unsqueeze_1134: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
    unsqueeze_1135: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
    add_359: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_425, unsqueeze_1135);  mul_425 = unsqueeze_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_138: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_359);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_138 = torch.ops.aten.split_with_sizes.default(relu_136, [104, 104, 104, 104], 1)
    getitem_556: "f32[8, 104, 14, 14]" = split_with_sizes_138[2];  split_with_sizes_138 = None
    add_360: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_138, getitem_556);  getitem_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_142: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_360, arg426_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_360 = arg426_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_284: "f32[104]" = torch.ops.prims.convert_element_type.default(arg938_1, torch.float32);  arg938_1 = None
    convert_element_type_285: "f32[104]" = torch.ops.prims.convert_element_type.default(arg939_1, torch.float32);  arg939_1 = None
    add_361: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_285, 1e-05);  convert_element_type_285 = None
    sqrt_142: "f32[104]" = torch.ops.aten.sqrt.default(add_361);  add_361 = None
    reciprocal_142: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
    mul_426: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
    unsqueeze_1136: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_284, -1);  convert_element_type_284 = None
    unsqueeze_1137: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
    unsqueeze_1138: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_426, -1);  mul_426 = None
    unsqueeze_1139: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
    sub_142: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_1137);  convolution_142 = unsqueeze_1137 = None
    mul_427: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
    unsqueeze_1140: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
    unsqueeze_1141: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
    mul_428: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_1141);  mul_427 = unsqueeze_1141 = None
    unsqueeze_1142: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
    unsqueeze_1143: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
    add_362: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_428, unsqueeze_1143);  mul_428 = unsqueeze_1143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_139: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_362);  add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_139 = torch.ops.aten.split_with_sizes.default(relu_136, [104, 104, 104, 104], 1);  relu_136 = None
    getitem_561: "f32[8, 104, 14, 14]" = split_with_sizes_139[3];  split_with_sizes_139 = None
    cat_27: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_137, relu_138, relu_139, getitem_561], 1);  relu_137 = relu_138 = relu_139 = getitem_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_143: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_27, arg429_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_27 = arg429_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_286: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg941_1, torch.float32);  arg941_1 = None
    convert_element_type_287: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg942_1, torch.float32);  arg942_1 = None
    add_363: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_287, 1e-05);  convert_element_type_287 = None
    sqrt_143: "f32[1024]" = torch.ops.aten.sqrt.default(add_363);  add_363 = None
    reciprocal_143: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
    mul_429: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
    unsqueeze_1144: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_286, -1);  convert_element_type_286 = None
    unsqueeze_1145: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
    unsqueeze_1146: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
    unsqueeze_1147: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
    sub_143: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_1145);  convolution_143 = unsqueeze_1145 = None
    mul_430: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
    unsqueeze_1148: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
    unsqueeze_1149: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
    mul_431: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1149);  mul_430 = unsqueeze_1149 = None
    unsqueeze_1150: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
    unsqueeze_1151: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
    add_364: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1151);  mul_431 = unsqueeze_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_365: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_364, relu_135);  add_364 = relu_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_140: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_365);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_144: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_140, arg432_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg432_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_288: "f32[416]" = torch.ops.prims.convert_element_type.default(arg944_1, torch.float32);  arg944_1 = None
    convert_element_type_289: "f32[416]" = torch.ops.prims.convert_element_type.default(arg945_1, torch.float32);  arg945_1 = None
    add_366: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_289, 1e-05);  convert_element_type_289 = None
    sqrt_144: "f32[416]" = torch.ops.aten.sqrt.default(add_366);  add_366 = None
    reciprocal_144: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
    mul_432: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
    unsqueeze_1152: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_288, -1);  convert_element_type_288 = None
    unsqueeze_1153: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
    unsqueeze_1154: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
    unsqueeze_1155: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
    sub_144: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1153);  convolution_144 = unsqueeze_1153 = None
    mul_433: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
    unsqueeze_1156: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg433_1, -1);  arg433_1 = None
    unsqueeze_1157: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
    mul_434: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1157);  mul_433 = unsqueeze_1157 = None
    unsqueeze_1158: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
    unsqueeze_1159: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
    add_367: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1159);  mul_434 = unsqueeze_1159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_141: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_367);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_141 = torch.ops.aten.split_with_sizes.default(relu_141, [104, 104, 104, 104], 1)
    getitem_566: "f32[8, 104, 14, 14]" = split_with_sizes_141[0];  split_with_sizes_141 = None
    convolution_145: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_566, arg435_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_566 = arg435_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_290: "f32[104]" = torch.ops.prims.convert_element_type.default(arg947_1, torch.float32);  arg947_1 = None
    convert_element_type_291: "f32[104]" = torch.ops.prims.convert_element_type.default(arg948_1, torch.float32);  arg948_1 = None
    add_368: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_291, 1e-05);  convert_element_type_291 = None
    sqrt_145: "f32[104]" = torch.ops.aten.sqrt.default(add_368);  add_368 = None
    reciprocal_145: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
    mul_435: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
    unsqueeze_1160: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_290, -1);  convert_element_type_290 = None
    unsqueeze_1161: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
    unsqueeze_1162: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
    unsqueeze_1163: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
    sub_145: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1161);  convolution_145 = unsqueeze_1161 = None
    mul_436: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
    unsqueeze_1164: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
    unsqueeze_1165: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
    mul_437: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1165);  mul_436 = unsqueeze_1165 = None
    unsqueeze_1166: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
    unsqueeze_1167: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
    add_369: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1167);  mul_437 = unsqueeze_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_142: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_369);  add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_142 = torch.ops.aten.split_with_sizes.default(relu_141, [104, 104, 104, 104], 1)
    getitem_571: "f32[8, 104, 14, 14]" = split_with_sizes_142[1];  split_with_sizes_142 = None
    add_370: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_142, getitem_571);  getitem_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_146: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_370, arg438_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_370 = arg438_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_292: "f32[104]" = torch.ops.prims.convert_element_type.default(arg950_1, torch.float32);  arg950_1 = None
    convert_element_type_293: "f32[104]" = torch.ops.prims.convert_element_type.default(arg951_1, torch.float32);  arg951_1 = None
    add_371: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_293, 1e-05);  convert_element_type_293 = None
    sqrt_146: "f32[104]" = torch.ops.aten.sqrt.default(add_371);  add_371 = None
    reciprocal_146: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
    mul_438: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
    unsqueeze_1168: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_292, -1);  convert_element_type_292 = None
    unsqueeze_1169: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
    unsqueeze_1170: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
    unsqueeze_1171: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
    sub_146: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_1169);  convolution_146 = unsqueeze_1169 = None
    mul_439: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
    unsqueeze_1172: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
    unsqueeze_1173: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
    mul_440: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1173);  mul_439 = unsqueeze_1173 = None
    unsqueeze_1174: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
    unsqueeze_1175: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
    add_372: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1175);  mul_440 = unsqueeze_1175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_143: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_372);  add_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_143 = torch.ops.aten.split_with_sizes.default(relu_141, [104, 104, 104, 104], 1)
    getitem_576: "f32[8, 104, 14, 14]" = split_with_sizes_143[2];  split_with_sizes_143 = None
    add_373: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_143, getitem_576);  getitem_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_147: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_373, arg441_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_373 = arg441_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_294: "f32[104]" = torch.ops.prims.convert_element_type.default(arg953_1, torch.float32);  arg953_1 = None
    convert_element_type_295: "f32[104]" = torch.ops.prims.convert_element_type.default(arg954_1, torch.float32);  arg954_1 = None
    add_374: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_295, 1e-05);  convert_element_type_295 = None
    sqrt_147: "f32[104]" = torch.ops.aten.sqrt.default(add_374);  add_374 = None
    reciprocal_147: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
    mul_441: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
    unsqueeze_1176: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_294, -1);  convert_element_type_294 = None
    unsqueeze_1177: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
    unsqueeze_1178: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
    unsqueeze_1179: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
    sub_147: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_1177);  convolution_147 = unsqueeze_1177 = None
    mul_442: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
    unsqueeze_1180: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
    unsqueeze_1181: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
    mul_443: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1181);  mul_442 = unsqueeze_1181 = None
    unsqueeze_1182: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
    unsqueeze_1183: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
    add_375: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1183);  mul_443 = unsqueeze_1183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_144: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_375);  add_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_144 = torch.ops.aten.split_with_sizes.default(relu_141, [104, 104, 104, 104], 1);  relu_141 = None
    getitem_581: "f32[8, 104, 14, 14]" = split_with_sizes_144[3];  split_with_sizes_144 = None
    cat_28: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_142, relu_143, relu_144, getitem_581], 1);  relu_142 = relu_143 = relu_144 = getitem_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_148: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_28, arg444_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_28 = arg444_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_296: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg956_1, torch.float32);  arg956_1 = None
    convert_element_type_297: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg957_1, torch.float32);  arg957_1 = None
    add_376: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_297, 1e-05);  convert_element_type_297 = None
    sqrt_148: "f32[1024]" = torch.ops.aten.sqrt.default(add_376);  add_376 = None
    reciprocal_148: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
    mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
    unsqueeze_1184: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_296, -1);  convert_element_type_296 = None
    unsqueeze_1185: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
    unsqueeze_1186: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
    unsqueeze_1187: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
    sub_148: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_1185);  convolution_148 = unsqueeze_1185 = None
    mul_445: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
    unsqueeze_1188: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
    unsqueeze_1189: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
    mul_446: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1189);  mul_445 = unsqueeze_1189 = None
    unsqueeze_1190: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
    unsqueeze_1191: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
    add_377: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1191);  mul_446 = unsqueeze_1191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_378: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_377, relu_140);  add_377 = relu_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_145: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_378);  add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_149: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_145, arg447_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg447_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_298: "f32[416]" = torch.ops.prims.convert_element_type.default(arg959_1, torch.float32);  arg959_1 = None
    convert_element_type_299: "f32[416]" = torch.ops.prims.convert_element_type.default(arg960_1, torch.float32);  arg960_1 = None
    add_379: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_299, 1e-05);  convert_element_type_299 = None
    sqrt_149: "f32[416]" = torch.ops.aten.sqrt.default(add_379);  add_379 = None
    reciprocal_149: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
    mul_447: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
    unsqueeze_1192: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_298, -1);  convert_element_type_298 = None
    unsqueeze_1193: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
    unsqueeze_1194: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
    unsqueeze_1195: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
    sub_149: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_1193);  convolution_149 = unsqueeze_1193 = None
    mul_448: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
    unsqueeze_1196: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
    unsqueeze_1197: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
    mul_449: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1197);  mul_448 = unsqueeze_1197 = None
    unsqueeze_1198: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
    unsqueeze_1199: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
    add_380: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1199);  mul_449 = unsqueeze_1199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_146: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_380);  add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_146 = torch.ops.aten.split_with_sizes.default(relu_146, [104, 104, 104, 104], 1)
    getitem_586: "f32[8, 104, 14, 14]" = split_with_sizes_146[0];  split_with_sizes_146 = None
    convolution_150: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_586, arg450_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_586 = arg450_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_300: "f32[104]" = torch.ops.prims.convert_element_type.default(arg962_1, torch.float32);  arg962_1 = None
    convert_element_type_301: "f32[104]" = torch.ops.prims.convert_element_type.default(arg963_1, torch.float32);  arg963_1 = None
    add_381: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_301, 1e-05);  convert_element_type_301 = None
    sqrt_150: "f32[104]" = torch.ops.aten.sqrt.default(add_381);  add_381 = None
    reciprocal_150: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
    mul_450: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
    unsqueeze_1200: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_300, -1);  convert_element_type_300 = None
    unsqueeze_1201: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
    unsqueeze_1202: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
    unsqueeze_1203: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
    sub_150: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_1201);  convolution_150 = unsqueeze_1201 = None
    mul_451: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
    unsqueeze_1204: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg451_1, -1);  arg451_1 = None
    unsqueeze_1205: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
    mul_452: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1205);  mul_451 = unsqueeze_1205 = None
    unsqueeze_1206: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
    unsqueeze_1207: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
    add_382: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1207);  mul_452 = unsqueeze_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_147: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_382);  add_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_147 = torch.ops.aten.split_with_sizes.default(relu_146, [104, 104, 104, 104], 1)
    getitem_591: "f32[8, 104, 14, 14]" = split_with_sizes_147[1];  split_with_sizes_147 = None
    add_383: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_147, getitem_591);  getitem_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_151: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_383, arg453_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_383 = arg453_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_302: "f32[104]" = torch.ops.prims.convert_element_type.default(arg965_1, torch.float32);  arg965_1 = None
    convert_element_type_303: "f32[104]" = torch.ops.prims.convert_element_type.default(arg966_1, torch.float32);  arg966_1 = None
    add_384: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_303, 1e-05);  convert_element_type_303 = None
    sqrt_151: "f32[104]" = torch.ops.aten.sqrt.default(add_384);  add_384 = None
    reciprocal_151: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
    mul_453: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
    unsqueeze_1208: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_302, -1);  convert_element_type_302 = None
    unsqueeze_1209: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
    unsqueeze_1210: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
    unsqueeze_1211: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
    sub_151: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1209);  convolution_151 = unsqueeze_1209 = None
    mul_454: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
    unsqueeze_1212: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
    unsqueeze_1213: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
    mul_455: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1213);  mul_454 = unsqueeze_1213 = None
    unsqueeze_1214: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
    unsqueeze_1215: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
    add_385: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1215);  mul_455 = unsqueeze_1215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_148: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_385);  add_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_148 = torch.ops.aten.split_with_sizes.default(relu_146, [104, 104, 104, 104], 1)
    getitem_596: "f32[8, 104, 14, 14]" = split_with_sizes_148[2];  split_with_sizes_148 = None
    add_386: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_148, getitem_596);  getitem_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_152: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_386, arg456_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_386 = arg456_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_304: "f32[104]" = torch.ops.prims.convert_element_type.default(arg968_1, torch.float32);  arg968_1 = None
    convert_element_type_305: "f32[104]" = torch.ops.prims.convert_element_type.default(arg969_1, torch.float32);  arg969_1 = None
    add_387: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_305, 1e-05);  convert_element_type_305 = None
    sqrt_152: "f32[104]" = torch.ops.aten.sqrt.default(add_387);  add_387 = None
    reciprocal_152: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
    mul_456: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
    unsqueeze_1216: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_304, -1);  convert_element_type_304 = None
    unsqueeze_1217: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
    unsqueeze_1218: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
    unsqueeze_1219: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
    sub_152: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1217);  convolution_152 = unsqueeze_1217 = None
    mul_457: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
    unsqueeze_1220: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
    unsqueeze_1221: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
    mul_458: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1221);  mul_457 = unsqueeze_1221 = None
    unsqueeze_1222: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg458_1, -1);  arg458_1 = None
    unsqueeze_1223: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
    add_388: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1223);  mul_458 = unsqueeze_1223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_149: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_388);  add_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_149 = torch.ops.aten.split_with_sizes.default(relu_146, [104, 104, 104, 104], 1);  relu_146 = None
    getitem_601: "f32[8, 104, 14, 14]" = split_with_sizes_149[3];  split_with_sizes_149 = None
    cat_29: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_147, relu_148, relu_149, getitem_601], 1);  relu_147 = relu_148 = relu_149 = getitem_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_153: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_29, arg459_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_29 = arg459_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_306: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg971_1, torch.float32);  arg971_1 = None
    convert_element_type_307: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg972_1, torch.float32);  arg972_1 = None
    add_389: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_307, 1e-05);  convert_element_type_307 = None
    sqrt_153: "f32[1024]" = torch.ops.aten.sqrt.default(add_389);  add_389 = None
    reciprocal_153: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
    mul_459: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
    unsqueeze_1224: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_306, -1);  convert_element_type_306 = None
    unsqueeze_1225: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
    unsqueeze_1226: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
    unsqueeze_1227: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
    sub_153: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_1225);  convolution_153 = unsqueeze_1225 = None
    mul_460: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
    unsqueeze_1228: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
    unsqueeze_1229: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
    mul_461: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1229);  mul_460 = unsqueeze_1229 = None
    unsqueeze_1230: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg461_1, -1);  arg461_1 = None
    unsqueeze_1231: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
    add_390: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1231);  mul_461 = unsqueeze_1231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_391: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_390, relu_145);  add_390 = relu_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_150: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_391);  add_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_154: "f32[8, 832, 14, 14]" = torch.ops.aten.convolution.default(relu_150, arg462_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg462_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_308: "f32[832]" = torch.ops.prims.convert_element_type.default(arg974_1, torch.float32);  arg974_1 = None
    convert_element_type_309: "f32[832]" = torch.ops.prims.convert_element_type.default(arg975_1, torch.float32);  arg975_1 = None
    add_392: "f32[832]" = torch.ops.aten.add.Tensor(convert_element_type_309, 1e-05);  convert_element_type_309 = None
    sqrt_154: "f32[832]" = torch.ops.aten.sqrt.default(add_392);  add_392 = None
    reciprocal_154: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
    mul_462: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
    unsqueeze_1232: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_308, -1);  convert_element_type_308 = None
    unsqueeze_1233: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
    unsqueeze_1234: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
    unsqueeze_1235: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
    sub_154: "f32[8, 832, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1233);  convolution_154 = unsqueeze_1233 = None
    mul_463: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
    unsqueeze_1236: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg463_1, -1);  arg463_1 = None
    unsqueeze_1237: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
    mul_464: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1237);  mul_463 = unsqueeze_1237 = None
    unsqueeze_1238: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
    unsqueeze_1239: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
    add_393: "f32[8, 832, 14, 14]" = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1239);  mul_464 = unsqueeze_1239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_151: "f32[8, 832, 14, 14]" = torch.ops.aten.relu.default(add_393);  add_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_151 = torch.ops.aten.split_with_sizes.default(relu_151, [208, 208, 208, 208], 1)
    getitem_606: "f32[8, 208, 14, 14]" = split_with_sizes_151[0];  split_with_sizes_151 = None
    convolution_155: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_606, arg465_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_606 = arg465_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_310: "f32[208]" = torch.ops.prims.convert_element_type.default(arg977_1, torch.float32);  arg977_1 = None
    convert_element_type_311: "f32[208]" = torch.ops.prims.convert_element_type.default(arg978_1, torch.float32);  arg978_1 = None
    add_394: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_311, 1e-05);  convert_element_type_311 = None
    sqrt_155: "f32[208]" = torch.ops.aten.sqrt.default(add_394);  add_394 = None
    reciprocal_155: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
    mul_465: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
    unsqueeze_1240: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_310, -1);  convert_element_type_310 = None
    unsqueeze_1241: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
    unsqueeze_1242: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_465, -1);  mul_465 = None
    unsqueeze_1243: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
    sub_155: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_1241);  convolution_155 = unsqueeze_1241 = None
    mul_466: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
    unsqueeze_1244: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
    unsqueeze_1245: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
    mul_467: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_1245);  mul_466 = unsqueeze_1245 = None
    unsqueeze_1246: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
    unsqueeze_1247: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
    add_395: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_467, unsqueeze_1247);  mul_467 = unsqueeze_1247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_152: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_395);  add_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_152 = torch.ops.aten.split_with_sizes.default(relu_151, [208, 208, 208, 208], 1)
    getitem_611: "f32[8, 208, 14, 14]" = split_with_sizes_152[1];  split_with_sizes_152 = None
    convolution_156: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_611, arg468_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_611 = arg468_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_312: "f32[208]" = torch.ops.prims.convert_element_type.default(arg980_1, torch.float32);  arg980_1 = None
    convert_element_type_313: "f32[208]" = torch.ops.prims.convert_element_type.default(arg981_1, torch.float32);  arg981_1 = None
    add_396: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_313, 1e-05);  convert_element_type_313 = None
    sqrt_156: "f32[208]" = torch.ops.aten.sqrt.default(add_396);  add_396 = None
    reciprocal_156: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
    mul_468: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
    unsqueeze_1248: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_312, -1);  convert_element_type_312 = None
    unsqueeze_1249: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
    unsqueeze_1250: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_468, -1);  mul_468 = None
    unsqueeze_1251: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
    sub_156: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_1249);  convolution_156 = unsqueeze_1249 = None
    mul_469: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
    unsqueeze_1252: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
    unsqueeze_1253: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
    mul_470: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_1253);  mul_469 = unsqueeze_1253 = None
    unsqueeze_1254: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
    unsqueeze_1255: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
    add_397: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_470, unsqueeze_1255);  mul_470 = unsqueeze_1255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_153: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_397);  add_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_153 = torch.ops.aten.split_with_sizes.default(relu_151, [208, 208, 208, 208], 1)
    getitem_616: "f32[8, 208, 14, 14]" = split_with_sizes_153[2];  split_with_sizes_153 = None
    convolution_157: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_616, arg471_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_616 = arg471_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_314: "f32[208]" = torch.ops.prims.convert_element_type.default(arg983_1, torch.float32);  arg983_1 = None
    convert_element_type_315: "f32[208]" = torch.ops.prims.convert_element_type.default(arg984_1, torch.float32);  arg984_1 = None
    add_398: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_315, 1e-05);  convert_element_type_315 = None
    sqrt_157: "f32[208]" = torch.ops.aten.sqrt.default(add_398);  add_398 = None
    reciprocal_157: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
    mul_471: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
    unsqueeze_1256: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_314, -1);  convert_element_type_314 = None
    unsqueeze_1257: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
    unsqueeze_1258: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_471, -1);  mul_471 = None
    unsqueeze_1259: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
    sub_157: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_1257);  convolution_157 = unsqueeze_1257 = None
    mul_472: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
    unsqueeze_1260: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
    unsqueeze_1261: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
    mul_473: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_1261);  mul_472 = unsqueeze_1261 = None
    unsqueeze_1262: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
    unsqueeze_1263: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
    add_399: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_473, unsqueeze_1263);  mul_473 = unsqueeze_1263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_154: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_399);  add_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_154 = torch.ops.aten.split_with_sizes.default(relu_151, [208, 208, 208, 208], 1);  relu_151 = None
    getitem_621: "f32[8, 208, 14, 14]" = split_with_sizes_154[3];  split_with_sizes_154 = None
    avg_pool2d_3: "f32[8, 208, 7, 7]" = torch.ops.aten.avg_pool2d.default(getitem_621, [3, 3], [2, 2], [1, 1]);  getitem_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_30: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_152, relu_153, relu_154, avg_pool2d_3], 1);  relu_152 = relu_153 = relu_154 = avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_158: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_30, arg474_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_30 = arg474_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_316: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg986_1, torch.float32);  arg986_1 = None
    convert_element_type_317: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg987_1, torch.float32);  arg987_1 = None
    add_400: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_317, 1e-05);  convert_element_type_317 = None
    sqrt_158: "f32[2048]" = torch.ops.aten.sqrt.default(add_400);  add_400 = None
    reciprocal_158: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
    mul_474: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
    unsqueeze_1264: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_316, -1);  convert_element_type_316 = None
    unsqueeze_1265: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
    unsqueeze_1266: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
    unsqueeze_1267: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
    sub_158: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_1265);  convolution_158 = unsqueeze_1265 = None
    mul_475: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
    unsqueeze_1268: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg475_1, -1);  arg475_1 = None
    unsqueeze_1269: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
    mul_476: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_1269);  mul_475 = unsqueeze_1269 = None
    unsqueeze_1270: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
    unsqueeze_1271: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
    add_401: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_476, unsqueeze_1271);  mul_476 = unsqueeze_1271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_159: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_150, arg477_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_150 = arg477_1 = None
    convert_element_type_318: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg989_1, torch.float32);  arg989_1 = None
    convert_element_type_319: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg990_1, torch.float32);  arg990_1 = None
    add_402: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_319, 1e-05);  convert_element_type_319 = None
    sqrt_159: "f32[2048]" = torch.ops.aten.sqrt.default(add_402);  add_402 = None
    reciprocal_159: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
    mul_477: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
    unsqueeze_1272: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_318, -1);  convert_element_type_318 = None
    unsqueeze_1273: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
    unsqueeze_1274: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_477, -1);  mul_477 = None
    unsqueeze_1275: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
    sub_159: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_159, unsqueeze_1273);  convolution_159 = unsqueeze_1273 = None
    mul_478: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
    unsqueeze_1276: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg478_1, -1);  arg478_1 = None
    unsqueeze_1277: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
    mul_479: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_478, unsqueeze_1277);  mul_478 = unsqueeze_1277 = None
    unsqueeze_1278: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
    unsqueeze_1279: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
    add_403: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_479, unsqueeze_1279);  mul_479 = unsqueeze_1279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_404: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_401, add_403);  add_401 = add_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_155: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_404);  add_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_160: "f32[8, 832, 7, 7]" = torch.ops.aten.convolution.default(relu_155, arg480_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg480_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_320: "f32[832]" = torch.ops.prims.convert_element_type.default(arg992_1, torch.float32);  arg992_1 = None
    convert_element_type_321: "f32[832]" = torch.ops.prims.convert_element_type.default(arg993_1, torch.float32);  arg993_1 = None
    add_405: "f32[832]" = torch.ops.aten.add.Tensor(convert_element_type_321, 1e-05);  convert_element_type_321 = None
    sqrt_160: "f32[832]" = torch.ops.aten.sqrt.default(add_405);  add_405 = None
    reciprocal_160: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
    mul_480: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
    unsqueeze_1280: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_320, -1);  convert_element_type_320 = None
    unsqueeze_1281: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
    unsqueeze_1282: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
    unsqueeze_1283: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
    sub_160: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_1281);  convolution_160 = unsqueeze_1281 = None
    mul_481: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_1283);  sub_160 = unsqueeze_1283 = None
    unsqueeze_1284: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg481_1, -1);  arg481_1 = None
    unsqueeze_1285: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
    mul_482: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_1285);  mul_481 = unsqueeze_1285 = None
    unsqueeze_1286: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
    unsqueeze_1287: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
    add_406: "f32[8, 832, 7, 7]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_1287);  mul_482 = unsqueeze_1287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_156: "f32[8, 832, 7, 7]" = torch.ops.aten.relu.default(add_406);  add_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_156 = torch.ops.aten.split_with_sizes.default(relu_156, [208, 208, 208, 208], 1)
    getitem_626: "f32[8, 208, 7, 7]" = split_with_sizes_156[0];  split_with_sizes_156 = None
    convolution_161: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_626, arg483_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_626 = arg483_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_322: "f32[208]" = torch.ops.prims.convert_element_type.default(arg995_1, torch.float32);  arg995_1 = None
    convert_element_type_323: "f32[208]" = torch.ops.prims.convert_element_type.default(arg996_1, torch.float32);  arg996_1 = None
    add_407: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_323, 1e-05);  convert_element_type_323 = None
    sqrt_161: "f32[208]" = torch.ops.aten.sqrt.default(add_407);  add_407 = None
    reciprocal_161: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
    mul_483: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
    unsqueeze_1288: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_322, -1);  convert_element_type_322 = None
    unsqueeze_1289: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
    unsqueeze_1290: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
    unsqueeze_1291: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
    sub_161: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_1289);  convolution_161 = unsqueeze_1289 = None
    mul_484: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_1291);  sub_161 = unsqueeze_1291 = None
    unsqueeze_1292: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
    unsqueeze_1293: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
    mul_485: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_1293);  mul_484 = unsqueeze_1293 = None
    unsqueeze_1294: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
    unsqueeze_1295: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
    add_408: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_485, unsqueeze_1295);  mul_485 = unsqueeze_1295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_157: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_408);  add_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_157 = torch.ops.aten.split_with_sizes.default(relu_156, [208, 208, 208, 208], 1)
    getitem_631: "f32[8, 208, 7, 7]" = split_with_sizes_157[1];  split_with_sizes_157 = None
    add_409: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_157, getitem_631);  getitem_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_162: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_409, arg486_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_409 = arg486_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_324: "f32[208]" = torch.ops.prims.convert_element_type.default(arg998_1, torch.float32);  arg998_1 = None
    convert_element_type_325: "f32[208]" = torch.ops.prims.convert_element_type.default(arg999_1, torch.float32);  arg999_1 = None
    add_410: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_325, 1e-05);  convert_element_type_325 = None
    sqrt_162: "f32[208]" = torch.ops.aten.sqrt.default(add_410);  add_410 = None
    reciprocal_162: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
    mul_486: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
    unsqueeze_1296: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_324, -1);  convert_element_type_324 = None
    unsqueeze_1297: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
    unsqueeze_1298: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_486, -1);  mul_486 = None
    unsqueeze_1299: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
    sub_162: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_1297);  convolution_162 = unsqueeze_1297 = None
    mul_487: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_1299);  sub_162 = unsqueeze_1299 = None
    unsqueeze_1300: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
    unsqueeze_1301: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
    mul_488: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_487, unsqueeze_1301);  mul_487 = unsqueeze_1301 = None
    unsqueeze_1302: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
    unsqueeze_1303: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
    add_411: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_488, unsqueeze_1303);  mul_488 = unsqueeze_1303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_158: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_411);  add_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_158 = torch.ops.aten.split_with_sizes.default(relu_156, [208, 208, 208, 208], 1)
    getitem_636: "f32[8, 208, 7, 7]" = split_with_sizes_158[2];  split_with_sizes_158 = None
    add_412: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_158, getitem_636);  getitem_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_163: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_412, arg489_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_412 = arg489_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_326: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1001_1, torch.float32);  arg1001_1 = None
    convert_element_type_327: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1002_1, torch.float32);  arg1002_1 = None
    add_413: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_327, 1e-05);  convert_element_type_327 = None
    sqrt_163: "f32[208]" = torch.ops.aten.sqrt.default(add_413);  add_413 = None
    reciprocal_163: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
    mul_489: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
    unsqueeze_1304: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_326, -1);  convert_element_type_326 = None
    unsqueeze_1305: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
    unsqueeze_1306: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_489, -1);  mul_489 = None
    unsqueeze_1307: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
    sub_163: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_163, unsqueeze_1305);  convolution_163 = unsqueeze_1305 = None
    mul_490: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_1307);  sub_163 = unsqueeze_1307 = None
    unsqueeze_1308: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
    unsqueeze_1309: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
    mul_491: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_1309);  mul_490 = unsqueeze_1309 = None
    unsqueeze_1310: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg491_1, -1);  arg491_1 = None
    unsqueeze_1311: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
    add_414: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_491, unsqueeze_1311);  mul_491 = unsqueeze_1311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_159: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_414);  add_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_159 = torch.ops.aten.split_with_sizes.default(relu_156, [208, 208, 208, 208], 1);  relu_156 = None
    getitem_641: "f32[8, 208, 7, 7]" = split_with_sizes_159[3];  split_with_sizes_159 = None
    cat_31: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_157, relu_158, relu_159, getitem_641], 1);  relu_157 = relu_158 = relu_159 = getitem_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_164: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_31, arg492_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_31 = arg492_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_328: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg1004_1, torch.float32);  arg1004_1 = None
    convert_element_type_329: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg1005_1, torch.float32);  arg1005_1 = None
    add_415: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_329, 1e-05);  convert_element_type_329 = None
    sqrt_164: "f32[2048]" = torch.ops.aten.sqrt.default(add_415);  add_415 = None
    reciprocal_164: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
    mul_492: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
    unsqueeze_1312: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_328, -1);  convert_element_type_328 = None
    unsqueeze_1313: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
    unsqueeze_1314: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_492, -1);  mul_492 = None
    unsqueeze_1315: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
    sub_164: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_1313);  convolution_164 = unsqueeze_1313 = None
    mul_493: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_1315);  sub_164 = unsqueeze_1315 = None
    unsqueeze_1316: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg493_1, -1);  arg493_1 = None
    unsqueeze_1317: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
    mul_494: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_1317);  mul_493 = unsqueeze_1317 = None
    unsqueeze_1318: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
    unsqueeze_1319: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
    add_416: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_494, unsqueeze_1319);  mul_494 = unsqueeze_1319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_417: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_416, relu_155);  add_416 = relu_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_160: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_417);  add_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_165: "f32[8, 832, 7, 7]" = torch.ops.aten.convolution.default(relu_160, arg495_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg495_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    convert_element_type_330: "f32[832]" = torch.ops.prims.convert_element_type.default(arg1007_1, torch.float32);  arg1007_1 = None
    convert_element_type_331: "f32[832]" = torch.ops.prims.convert_element_type.default(arg1008_1, torch.float32);  arg1008_1 = None
    add_418: "f32[832]" = torch.ops.aten.add.Tensor(convert_element_type_331, 1e-05);  convert_element_type_331 = None
    sqrt_165: "f32[832]" = torch.ops.aten.sqrt.default(add_418);  add_418 = None
    reciprocal_165: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
    mul_495: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
    unsqueeze_1320: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_330, -1);  convert_element_type_330 = None
    unsqueeze_1321: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
    unsqueeze_1322: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
    unsqueeze_1323: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
    sub_165: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_1321);  convolution_165 = unsqueeze_1321 = None
    mul_496: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_1323);  sub_165 = unsqueeze_1323 = None
    unsqueeze_1324: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg496_1, -1);  arg496_1 = None
    unsqueeze_1325: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
    mul_497: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_1325);  mul_496 = unsqueeze_1325 = None
    unsqueeze_1326: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
    unsqueeze_1327: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
    add_419: "f32[8, 832, 7, 7]" = torch.ops.aten.add.Tensor(mul_497, unsqueeze_1327);  mul_497 = unsqueeze_1327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_161: "f32[8, 832, 7, 7]" = torch.ops.aten.relu.default(add_419);  add_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_161 = torch.ops.aten.split_with_sizes.default(relu_161, [208, 208, 208, 208], 1)
    getitem_646: "f32[8, 208, 7, 7]" = split_with_sizes_161[0];  split_with_sizes_161 = None
    convolution_166: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_646, arg498_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_646 = arg498_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_332: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1010_1, torch.float32);  arg1010_1 = None
    convert_element_type_333: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1011_1, torch.float32);  arg1011_1 = None
    add_420: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_333, 1e-05);  convert_element_type_333 = None
    sqrt_166: "f32[208]" = torch.ops.aten.sqrt.default(add_420);  add_420 = None
    reciprocal_166: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
    mul_498: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
    unsqueeze_1328: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_332, -1);  convert_element_type_332 = None
    unsqueeze_1329: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
    unsqueeze_1330: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
    unsqueeze_1331: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
    sub_166: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_1329);  convolution_166 = unsqueeze_1329 = None
    mul_499: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_1331);  sub_166 = unsqueeze_1331 = None
    unsqueeze_1332: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
    unsqueeze_1333: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
    mul_500: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_1333);  mul_499 = unsqueeze_1333 = None
    unsqueeze_1334: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
    unsqueeze_1335: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
    add_421: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_500, unsqueeze_1335);  mul_500 = unsqueeze_1335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_162: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_421);  add_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_162 = torch.ops.aten.split_with_sizes.default(relu_161, [208, 208, 208, 208], 1)
    getitem_651: "f32[8, 208, 7, 7]" = split_with_sizes_162[1];  split_with_sizes_162 = None
    add_422: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_162, getitem_651);  getitem_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_167: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_422, arg501_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_422 = arg501_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_334: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1013_1, torch.float32);  arg1013_1 = None
    convert_element_type_335: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1014_1, torch.float32);  arg1014_1 = None
    add_423: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_335, 1e-05);  convert_element_type_335 = None
    sqrt_167: "f32[208]" = torch.ops.aten.sqrt.default(add_423);  add_423 = None
    reciprocal_167: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
    mul_501: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
    unsqueeze_1336: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_334, -1);  convert_element_type_334 = None
    unsqueeze_1337: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
    unsqueeze_1338: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_501, -1);  mul_501 = None
    unsqueeze_1339: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
    sub_167: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_1337);  convolution_167 = unsqueeze_1337 = None
    mul_502: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_1339);  sub_167 = unsqueeze_1339 = None
    unsqueeze_1340: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
    unsqueeze_1341: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
    mul_503: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_1341);  mul_502 = unsqueeze_1341 = None
    unsqueeze_1342: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
    unsqueeze_1343: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
    add_424: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_1343);  mul_503 = unsqueeze_1343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_163: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_424);  add_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_163 = torch.ops.aten.split_with_sizes.default(relu_161, [208, 208, 208, 208], 1)
    getitem_656: "f32[8, 208, 7, 7]" = split_with_sizes_163[2];  split_with_sizes_163 = None
    add_425: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_163, getitem_656);  getitem_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_168: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_425, arg504_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_425 = arg504_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    convert_element_type_336: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1016_1, torch.float32);  arg1016_1 = None
    convert_element_type_337: "f32[208]" = torch.ops.prims.convert_element_type.default(arg1017_1, torch.float32);  arg1017_1 = None
    add_426: "f32[208]" = torch.ops.aten.add.Tensor(convert_element_type_337, 1e-05);  convert_element_type_337 = None
    sqrt_168: "f32[208]" = torch.ops.aten.sqrt.default(add_426);  add_426 = None
    reciprocal_168: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
    mul_504: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
    unsqueeze_1344: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_336, -1);  convert_element_type_336 = None
    unsqueeze_1345: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
    unsqueeze_1346: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
    unsqueeze_1347: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
    sub_168: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_1345);  convolution_168 = unsqueeze_1345 = None
    mul_505: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_1347);  sub_168 = unsqueeze_1347 = None
    unsqueeze_1348: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
    unsqueeze_1349: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
    mul_506: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1349);  mul_505 = unsqueeze_1349 = None
    unsqueeze_1350: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
    unsqueeze_1351: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
    add_427: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1351);  mul_506 = unsqueeze_1351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_164: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_427);  add_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_164 = torch.ops.aten.split_with_sizes.default(relu_161, [208, 208, 208, 208], 1);  relu_161 = None
    getitem_661: "f32[8, 208, 7, 7]" = split_with_sizes_164[3];  split_with_sizes_164 = None
    cat_32: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_162, relu_163, relu_164, getitem_661], 1);  relu_162 = relu_163 = relu_164 = getitem_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_169: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_32, arg507_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_32 = arg507_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    convert_element_type_338: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg1019_1, torch.float32);  arg1019_1 = None
    convert_element_type_339: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg1020_1, torch.float32);  arg1020_1 = None
    add_428: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_339, 1e-05);  convert_element_type_339 = None
    sqrt_169: "f32[2048]" = torch.ops.aten.sqrt.default(add_428);  add_428 = None
    reciprocal_169: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
    mul_507: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
    unsqueeze_1352: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_338, -1);  convert_element_type_338 = None
    unsqueeze_1353: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
    unsqueeze_1354: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
    unsqueeze_1355: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
    sub_169: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_1353);  convolution_169 = unsqueeze_1353 = None
    mul_508: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_1355);  sub_169 = unsqueeze_1355 = None
    unsqueeze_1356: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg508_1, -1);  arg508_1 = None
    unsqueeze_1357: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
    mul_509: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_1357);  mul_508 = unsqueeze_1357 = None
    unsqueeze_1358: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
    unsqueeze_1359: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
    add_429: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_509, unsqueeze_1359);  mul_509 = unsqueeze_1359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_430: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_429, relu_160);  add_429 = relu_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_165: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_430);  add_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_165, [-1, -2], True);  relu_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 2048]" = torch.ops.aten.view.default(mean, [8, 2048]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg511_1, view, permute);  arg511_1 = view = permute = None
    return (addmm,)
    