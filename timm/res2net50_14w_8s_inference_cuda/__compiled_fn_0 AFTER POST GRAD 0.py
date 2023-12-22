from __future__ import annotations



def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[112, 64, 1, 1]", arg4_1: "f32[112]", arg5_1: "f32[112]", arg6_1: "f32[14, 14, 3, 3]", arg7_1: "f32[14]", arg8_1: "f32[14]", arg9_1: "f32[14, 14, 3, 3]", arg10_1: "f32[14]", arg11_1: "f32[14]", arg12_1: "f32[14, 14, 3, 3]", arg13_1: "f32[14]", arg14_1: "f32[14]", arg15_1: "f32[14, 14, 3, 3]", arg16_1: "f32[14]", arg17_1: "f32[14]", arg18_1: "f32[14, 14, 3, 3]", arg19_1: "f32[14]", arg20_1: "f32[14]", arg21_1: "f32[14, 14, 3, 3]", arg22_1: "f32[14]", arg23_1: "f32[14]", arg24_1: "f32[14, 14, 3, 3]", arg25_1: "f32[14]", arg26_1: "f32[14]", arg27_1: "f32[256, 112, 1, 1]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256, 64, 1, 1]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[112, 256, 1, 1]", arg34_1: "f32[112]", arg35_1: "f32[112]", arg36_1: "f32[14, 14, 3, 3]", arg37_1: "f32[14]", arg38_1: "f32[14]", arg39_1: "f32[14, 14, 3, 3]", arg40_1: "f32[14]", arg41_1: "f32[14]", arg42_1: "f32[14, 14, 3, 3]", arg43_1: "f32[14]", arg44_1: "f32[14]", arg45_1: "f32[14, 14, 3, 3]", arg46_1: "f32[14]", arg47_1: "f32[14]", arg48_1: "f32[14, 14, 3, 3]", arg49_1: "f32[14]", arg50_1: "f32[14]", arg51_1: "f32[14, 14, 3, 3]", arg52_1: "f32[14]", arg53_1: "f32[14]", arg54_1: "f32[14, 14, 3, 3]", arg55_1: "f32[14]", arg56_1: "f32[14]", arg57_1: "f32[256, 112, 1, 1]", arg58_1: "f32[256]", arg59_1: "f32[256]", arg60_1: "f32[112, 256, 1, 1]", arg61_1: "f32[112]", arg62_1: "f32[112]", arg63_1: "f32[14, 14, 3, 3]", arg64_1: "f32[14]", arg65_1: "f32[14]", arg66_1: "f32[14, 14, 3, 3]", arg67_1: "f32[14]", arg68_1: "f32[14]", arg69_1: "f32[14, 14, 3, 3]", arg70_1: "f32[14]", arg71_1: "f32[14]", arg72_1: "f32[14, 14, 3, 3]", arg73_1: "f32[14]", arg74_1: "f32[14]", arg75_1: "f32[14, 14, 3, 3]", arg76_1: "f32[14]", arg77_1: "f32[14]", arg78_1: "f32[14, 14, 3, 3]", arg79_1: "f32[14]", arg80_1: "f32[14]", arg81_1: "f32[14, 14, 3, 3]", arg82_1: "f32[14]", arg83_1: "f32[14]", arg84_1: "f32[256, 112, 1, 1]", arg85_1: "f32[256]", arg86_1: "f32[256]", arg87_1: "f32[224, 256, 1, 1]", arg88_1: "f32[224]", arg89_1: "f32[224]", arg90_1: "f32[28, 28, 3, 3]", arg91_1: "f32[28]", arg92_1: "f32[28]", arg93_1: "f32[28, 28, 3, 3]", arg94_1: "f32[28]", arg95_1: "f32[28]", arg96_1: "f32[28, 28, 3, 3]", arg97_1: "f32[28]", arg98_1: "f32[28]", arg99_1: "f32[28, 28, 3, 3]", arg100_1: "f32[28]", arg101_1: "f32[28]", arg102_1: "f32[28, 28, 3, 3]", arg103_1: "f32[28]", arg104_1: "f32[28]", arg105_1: "f32[28, 28, 3, 3]", arg106_1: "f32[28]", arg107_1: "f32[28]", arg108_1: "f32[28, 28, 3, 3]", arg109_1: "f32[28]", arg110_1: "f32[28]", arg111_1: "f32[512, 224, 1, 1]", arg112_1: "f32[512]", arg113_1: "f32[512]", arg114_1: "f32[512, 256, 1, 1]", arg115_1: "f32[512]", arg116_1: "f32[512]", arg117_1: "f32[224, 512, 1, 1]", arg118_1: "f32[224]", arg119_1: "f32[224]", arg120_1: "f32[28, 28, 3, 3]", arg121_1: "f32[28]", arg122_1: "f32[28]", arg123_1: "f32[28, 28, 3, 3]", arg124_1: "f32[28]", arg125_1: "f32[28]", arg126_1: "f32[28, 28, 3, 3]", arg127_1: "f32[28]", arg128_1: "f32[28]", arg129_1: "f32[28, 28, 3, 3]", arg130_1: "f32[28]", arg131_1: "f32[28]", arg132_1: "f32[28, 28, 3, 3]", arg133_1: "f32[28]", arg134_1: "f32[28]", arg135_1: "f32[28, 28, 3, 3]", arg136_1: "f32[28]", arg137_1: "f32[28]", arg138_1: "f32[28, 28, 3, 3]", arg139_1: "f32[28]", arg140_1: "f32[28]", arg141_1: "f32[512, 224, 1, 1]", arg142_1: "f32[512]", arg143_1: "f32[512]", arg144_1: "f32[224, 512, 1, 1]", arg145_1: "f32[224]", arg146_1: "f32[224]", arg147_1: "f32[28, 28, 3, 3]", arg148_1: "f32[28]", arg149_1: "f32[28]", arg150_1: "f32[28, 28, 3, 3]", arg151_1: "f32[28]", arg152_1: "f32[28]", arg153_1: "f32[28, 28, 3, 3]", arg154_1: "f32[28]", arg155_1: "f32[28]", arg156_1: "f32[28, 28, 3, 3]", arg157_1: "f32[28]", arg158_1: "f32[28]", arg159_1: "f32[28, 28, 3, 3]", arg160_1: "f32[28]", arg161_1: "f32[28]", arg162_1: "f32[28, 28, 3, 3]", arg163_1: "f32[28]", arg164_1: "f32[28]", arg165_1: "f32[28, 28, 3, 3]", arg166_1: "f32[28]", arg167_1: "f32[28]", arg168_1: "f32[512, 224, 1, 1]", arg169_1: "f32[512]", arg170_1: "f32[512]", arg171_1: "f32[224, 512, 1, 1]", arg172_1: "f32[224]", arg173_1: "f32[224]", arg174_1: "f32[28, 28, 3, 3]", arg175_1: "f32[28]", arg176_1: "f32[28]", arg177_1: "f32[28, 28, 3, 3]", arg178_1: "f32[28]", arg179_1: "f32[28]", arg180_1: "f32[28, 28, 3, 3]", arg181_1: "f32[28]", arg182_1: "f32[28]", arg183_1: "f32[28, 28, 3, 3]", arg184_1: "f32[28]", arg185_1: "f32[28]", arg186_1: "f32[28, 28, 3, 3]", arg187_1: "f32[28]", arg188_1: "f32[28]", arg189_1: "f32[28, 28, 3, 3]", arg190_1: "f32[28]", arg191_1: "f32[28]", arg192_1: "f32[28, 28, 3, 3]", arg193_1: "f32[28]", arg194_1: "f32[28]", arg195_1: "f32[512, 224, 1, 1]", arg196_1: "f32[512]", arg197_1: "f32[512]", arg198_1: "f32[448, 512, 1, 1]", arg199_1: "f32[448]", arg200_1: "f32[448]", arg201_1: "f32[56, 56, 3, 3]", arg202_1: "f32[56]", arg203_1: "f32[56]", arg204_1: "f32[56, 56, 3, 3]", arg205_1: "f32[56]", arg206_1: "f32[56]", arg207_1: "f32[56, 56, 3, 3]", arg208_1: "f32[56]", arg209_1: "f32[56]", arg210_1: "f32[56, 56, 3, 3]", arg211_1: "f32[56]", arg212_1: "f32[56]", arg213_1: "f32[56, 56, 3, 3]", arg214_1: "f32[56]", arg215_1: "f32[56]", arg216_1: "f32[56, 56, 3, 3]", arg217_1: "f32[56]", arg218_1: "f32[56]", arg219_1: "f32[56, 56, 3, 3]", arg220_1: "f32[56]", arg221_1: "f32[56]", arg222_1: "f32[1024, 448, 1, 1]", arg223_1: "f32[1024]", arg224_1: "f32[1024]", arg225_1: "f32[1024, 512, 1, 1]", arg226_1: "f32[1024]", arg227_1: "f32[1024]", arg228_1: "f32[448, 1024, 1, 1]", arg229_1: "f32[448]", arg230_1: "f32[448]", arg231_1: "f32[56, 56, 3, 3]", arg232_1: "f32[56]", arg233_1: "f32[56]", arg234_1: "f32[56, 56, 3, 3]", arg235_1: "f32[56]", arg236_1: "f32[56]", arg237_1: "f32[56, 56, 3, 3]", arg238_1: "f32[56]", arg239_1: "f32[56]", arg240_1: "f32[56, 56, 3, 3]", arg241_1: "f32[56]", arg242_1: "f32[56]", arg243_1: "f32[56, 56, 3, 3]", arg244_1: "f32[56]", arg245_1: "f32[56]", arg246_1: "f32[56, 56, 3, 3]", arg247_1: "f32[56]", arg248_1: "f32[56]", arg249_1: "f32[56, 56, 3, 3]", arg250_1: "f32[56]", arg251_1: "f32[56]", arg252_1: "f32[1024, 448, 1, 1]", arg253_1: "f32[1024]", arg254_1: "f32[1024]", arg255_1: "f32[448, 1024, 1, 1]", arg256_1: "f32[448]", arg257_1: "f32[448]", arg258_1: "f32[56, 56, 3, 3]", arg259_1: "f32[56]", arg260_1: "f32[56]", arg261_1: "f32[56, 56, 3, 3]", arg262_1: "f32[56]", arg263_1: "f32[56]", arg264_1: "f32[56, 56, 3, 3]", arg265_1: "f32[56]", arg266_1: "f32[56]", arg267_1: "f32[56, 56, 3, 3]", arg268_1: "f32[56]", arg269_1: "f32[56]", arg270_1: "f32[56, 56, 3, 3]", arg271_1: "f32[56]", arg272_1: "f32[56]", arg273_1: "f32[56, 56, 3, 3]", arg274_1: "f32[56]", arg275_1: "f32[56]", arg276_1: "f32[56, 56, 3, 3]", arg277_1: "f32[56]", arg278_1: "f32[56]", arg279_1: "f32[1024, 448, 1, 1]", arg280_1: "f32[1024]", arg281_1: "f32[1024]", arg282_1: "f32[448, 1024, 1, 1]", arg283_1: "f32[448]", arg284_1: "f32[448]", arg285_1: "f32[56, 56, 3, 3]", arg286_1: "f32[56]", arg287_1: "f32[56]", arg288_1: "f32[56, 56, 3, 3]", arg289_1: "f32[56]", arg290_1: "f32[56]", arg291_1: "f32[56, 56, 3, 3]", arg292_1: "f32[56]", arg293_1: "f32[56]", arg294_1: "f32[56, 56, 3, 3]", arg295_1: "f32[56]", arg296_1: "f32[56]", arg297_1: "f32[56, 56, 3, 3]", arg298_1: "f32[56]", arg299_1: "f32[56]", arg300_1: "f32[56, 56, 3, 3]", arg301_1: "f32[56]", arg302_1: "f32[56]", arg303_1: "f32[56, 56, 3, 3]", arg304_1: "f32[56]", arg305_1: "f32[56]", arg306_1: "f32[1024, 448, 1, 1]", arg307_1: "f32[1024]", arg308_1: "f32[1024]", arg309_1: "f32[448, 1024, 1, 1]", arg310_1: "f32[448]", arg311_1: "f32[448]", arg312_1: "f32[56, 56, 3, 3]", arg313_1: "f32[56]", arg314_1: "f32[56]", arg315_1: "f32[56, 56, 3, 3]", arg316_1: "f32[56]", arg317_1: "f32[56]", arg318_1: "f32[56, 56, 3, 3]", arg319_1: "f32[56]", arg320_1: "f32[56]", arg321_1: "f32[56, 56, 3, 3]", arg322_1: "f32[56]", arg323_1: "f32[56]", arg324_1: "f32[56, 56, 3, 3]", arg325_1: "f32[56]", arg326_1: "f32[56]", arg327_1: "f32[56, 56, 3, 3]", arg328_1: "f32[56]", arg329_1: "f32[56]", arg330_1: "f32[56, 56, 3, 3]", arg331_1: "f32[56]", arg332_1: "f32[56]", arg333_1: "f32[1024, 448, 1, 1]", arg334_1: "f32[1024]", arg335_1: "f32[1024]", arg336_1: "f32[448, 1024, 1, 1]", arg337_1: "f32[448]", arg338_1: "f32[448]", arg339_1: "f32[56, 56, 3, 3]", arg340_1: "f32[56]", arg341_1: "f32[56]", arg342_1: "f32[56, 56, 3, 3]", arg343_1: "f32[56]", arg344_1: "f32[56]", arg345_1: "f32[56, 56, 3, 3]", arg346_1: "f32[56]", arg347_1: "f32[56]", arg348_1: "f32[56, 56, 3, 3]", arg349_1: "f32[56]", arg350_1: "f32[56]", arg351_1: "f32[56, 56, 3, 3]", arg352_1: "f32[56]", arg353_1: "f32[56]", arg354_1: "f32[56, 56, 3, 3]", arg355_1: "f32[56]", arg356_1: "f32[56]", arg357_1: "f32[56, 56, 3, 3]", arg358_1: "f32[56]", arg359_1: "f32[56]", arg360_1: "f32[1024, 448, 1, 1]", arg361_1: "f32[1024]", arg362_1: "f32[1024]", arg363_1: "f32[896, 1024, 1, 1]", arg364_1: "f32[896]", arg365_1: "f32[896]", arg366_1: "f32[112, 112, 3, 3]", arg367_1: "f32[112]", arg368_1: "f32[112]", arg369_1: "f32[112, 112, 3, 3]", arg370_1: "f32[112]", arg371_1: "f32[112]", arg372_1: "f32[112, 112, 3, 3]", arg373_1: "f32[112]", arg374_1: "f32[112]", arg375_1: "f32[112, 112, 3, 3]", arg376_1: "f32[112]", arg377_1: "f32[112]", arg378_1: "f32[112, 112, 3, 3]", arg379_1: "f32[112]", arg380_1: "f32[112]", arg381_1: "f32[112, 112, 3, 3]", arg382_1: "f32[112]", arg383_1: "f32[112]", arg384_1: "f32[112, 112, 3, 3]", arg385_1: "f32[112]", arg386_1: "f32[112]", arg387_1: "f32[2048, 896, 1, 1]", arg388_1: "f32[2048]", arg389_1: "f32[2048]", arg390_1: "f32[2048, 1024, 1, 1]", arg391_1: "f32[2048]", arg392_1: "f32[2048]", arg393_1: "f32[896, 2048, 1, 1]", arg394_1: "f32[896]", arg395_1: "f32[896]", arg396_1: "f32[112, 112, 3, 3]", arg397_1: "f32[112]", arg398_1: "f32[112]", arg399_1: "f32[112, 112, 3, 3]", arg400_1: "f32[112]", arg401_1: "f32[112]", arg402_1: "f32[112, 112, 3, 3]", arg403_1: "f32[112]", arg404_1: "f32[112]", arg405_1: "f32[112, 112, 3, 3]", arg406_1: "f32[112]", arg407_1: "f32[112]", arg408_1: "f32[112, 112, 3, 3]", arg409_1: "f32[112]", arg410_1: "f32[112]", arg411_1: "f32[112, 112, 3, 3]", arg412_1: "f32[112]", arg413_1: "f32[112]", arg414_1: "f32[112, 112, 3, 3]", arg415_1: "f32[112]", arg416_1: "f32[112]", arg417_1: "f32[2048, 896, 1, 1]", arg418_1: "f32[2048]", arg419_1: "f32[2048]", arg420_1: "f32[896, 2048, 1, 1]", arg421_1: "f32[896]", arg422_1: "f32[896]", arg423_1: "f32[112, 112, 3, 3]", arg424_1: "f32[112]", arg425_1: "f32[112]", arg426_1: "f32[112, 112, 3, 3]", arg427_1: "f32[112]", arg428_1: "f32[112]", arg429_1: "f32[112, 112, 3, 3]", arg430_1: "f32[112]", arg431_1: "f32[112]", arg432_1: "f32[112, 112, 3, 3]", arg433_1: "f32[112]", arg434_1: "f32[112]", arg435_1: "f32[112, 112, 3, 3]", arg436_1: "f32[112]", arg437_1: "f32[112]", arg438_1: "f32[112, 112, 3, 3]", arg439_1: "f32[112]", arg440_1: "f32[112]", arg441_1: "f32[112, 112, 3, 3]", arg442_1: "f32[112]", arg443_1: "f32[112]", arg444_1: "f32[2048, 896, 1, 1]", arg445_1: "f32[2048]", arg446_1: "f32[2048]", arg447_1: "f32[1000, 2048]", arg448_1: "f32[1000]", arg449_1: "f32[64]", arg450_1: "f32[64]", arg451_1: "i64[]", arg452_1: "f32[112]", arg453_1: "f32[112]", arg454_1: "i64[]", arg455_1: "f32[14]", arg456_1: "f32[14]", arg457_1: "i64[]", arg458_1: "f32[14]", arg459_1: "f32[14]", arg460_1: "i64[]", arg461_1: "f32[14]", arg462_1: "f32[14]", arg463_1: "i64[]", arg464_1: "f32[14]", arg465_1: "f32[14]", arg466_1: "i64[]", arg467_1: "f32[14]", arg468_1: "f32[14]", arg469_1: "i64[]", arg470_1: "f32[14]", arg471_1: "f32[14]", arg472_1: "i64[]", arg473_1: "f32[14]", arg474_1: "f32[14]", arg475_1: "i64[]", arg476_1: "f32[256]", arg477_1: "f32[256]", arg478_1: "i64[]", arg479_1: "f32[256]", arg480_1: "f32[256]", arg481_1: "i64[]", arg482_1: "f32[112]", arg483_1: "f32[112]", arg484_1: "i64[]", arg485_1: "f32[14]", arg486_1: "f32[14]", arg487_1: "i64[]", arg488_1: "f32[14]", arg489_1: "f32[14]", arg490_1: "i64[]", arg491_1: "f32[14]", arg492_1: "f32[14]", arg493_1: "i64[]", arg494_1: "f32[14]", arg495_1: "f32[14]", arg496_1: "i64[]", arg497_1: "f32[14]", arg498_1: "f32[14]", arg499_1: "i64[]", arg500_1: "f32[14]", arg501_1: "f32[14]", arg502_1: "i64[]", arg503_1: "f32[14]", arg504_1: "f32[14]", arg505_1: "i64[]", arg506_1: "f32[256]", arg507_1: "f32[256]", arg508_1: "i64[]", arg509_1: "f32[112]", arg510_1: "f32[112]", arg511_1: "i64[]", arg512_1: "f32[14]", arg513_1: "f32[14]", arg514_1: "i64[]", arg515_1: "f32[14]", arg516_1: "f32[14]", arg517_1: "i64[]", arg518_1: "f32[14]", arg519_1: "f32[14]", arg520_1: "i64[]", arg521_1: "f32[14]", arg522_1: "f32[14]", arg523_1: "i64[]", arg524_1: "f32[14]", arg525_1: "f32[14]", arg526_1: "i64[]", arg527_1: "f32[14]", arg528_1: "f32[14]", arg529_1: "i64[]", arg530_1: "f32[14]", arg531_1: "f32[14]", arg532_1: "i64[]", arg533_1: "f32[256]", arg534_1: "f32[256]", arg535_1: "i64[]", arg536_1: "f32[224]", arg537_1: "f32[224]", arg538_1: "i64[]", arg539_1: "f32[28]", arg540_1: "f32[28]", arg541_1: "i64[]", arg542_1: "f32[28]", arg543_1: "f32[28]", arg544_1: "i64[]", arg545_1: "f32[28]", arg546_1: "f32[28]", arg547_1: "i64[]", arg548_1: "f32[28]", arg549_1: "f32[28]", arg550_1: "i64[]", arg551_1: "f32[28]", arg552_1: "f32[28]", arg553_1: "i64[]", arg554_1: "f32[28]", arg555_1: "f32[28]", arg556_1: "i64[]", arg557_1: "f32[28]", arg558_1: "f32[28]", arg559_1: "i64[]", arg560_1: "f32[512]", arg561_1: "f32[512]", arg562_1: "i64[]", arg563_1: "f32[512]", arg564_1: "f32[512]", arg565_1: "i64[]", arg566_1: "f32[224]", arg567_1: "f32[224]", arg568_1: "i64[]", arg569_1: "f32[28]", arg570_1: "f32[28]", arg571_1: "i64[]", arg572_1: "f32[28]", arg573_1: "f32[28]", arg574_1: "i64[]", arg575_1: "f32[28]", arg576_1: "f32[28]", arg577_1: "i64[]", arg578_1: "f32[28]", arg579_1: "f32[28]", arg580_1: "i64[]", arg581_1: "f32[28]", arg582_1: "f32[28]", arg583_1: "i64[]", arg584_1: "f32[28]", arg585_1: "f32[28]", arg586_1: "i64[]", arg587_1: "f32[28]", arg588_1: "f32[28]", arg589_1: "i64[]", arg590_1: "f32[512]", arg591_1: "f32[512]", arg592_1: "i64[]", arg593_1: "f32[224]", arg594_1: "f32[224]", arg595_1: "i64[]", arg596_1: "f32[28]", arg597_1: "f32[28]", arg598_1: "i64[]", arg599_1: "f32[28]", arg600_1: "f32[28]", arg601_1: "i64[]", arg602_1: "f32[28]", arg603_1: "f32[28]", arg604_1: "i64[]", arg605_1: "f32[28]", arg606_1: "f32[28]", arg607_1: "i64[]", arg608_1: "f32[28]", arg609_1: "f32[28]", arg610_1: "i64[]", arg611_1: "f32[28]", arg612_1: "f32[28]", arg613_1: "i64[]", arg614_1: "f32[28]", arg615_1: "f32[28]", arg616_1: "i64[]", arg617_1: "f32[512]", arg618_1: "f32[512]", arg619_1: "i64[]", arg620_1: "f32[224]", arg621_1: "f32[224]", arg622_1: "i64[]", arg623_1: "f32[28]", arg624_1: "f32[28]", arg625_1: "i64[]", arg626_1: "f32[28]", arg627_1: "f32[28]", arg628_1: "i64[]", arg629_1: "f32[28]", arg630_1: "f32[28]", arg631_1: "i64[]", arg632_1: "f32[28]", arg633_1: "f32[28]", arg634_1: "i64[]", arg635_1: "f32[28]", arg636_1: "f32[28]", arg637_1: "i64[]", arg638_1: "f32[28]", arg639_1: "f32[28]", arg640_1: "i64[]", arg641_1: "f32[28]", arg642_1: "f32[28]", arg643_1: "i64[]", arg644_1: "f32[512]", arg645_1: "f32[512]", arg646_1: "i64[]", arg647_1: "f32[448]", arg648_1: "f32[448]", arg649_1: "i64[]", arg650_1: "f32[56]", arg651_1: "f32[56]", arg652_1: "i64[]", arg653_1: "f32[56]", arg654_1: "f32[56]", arg655_1: "i64[]", arg656_1: "f32[56]", arg657_1: "f32[56]", arg658_1: "i64[]", arg659_1: "f32[56]", arg660_1: "f32[56]", arg661_1: "i64[]", arg662_1: "f32[56]", arg663_1: "f32[56]", arg664_1: "i64[]", arg665_1: "f32[56]", arg666_1: "f32[56]", arg667_1: "i64[]", arg668_1: "f32[56]", arg669_1: "f32[56]", arg670_1: "i64[]", arg671_1: "f32[1024]", arg672_1: "f32[1024]", arg673_1: "i64[]", arg674_1: "f32[1024]", arg675_1: "f32[1024]", arg676_1: "i64[]", arg677_1: "f32[448]", arg678_1: "f32[448]", arg679_1: "i64[]", arg680_1: "f32[56]", arg681_1: "f32[56]", arg682_1: "i64[]", arg683_1: "f32[56]", arg684_1: "f32[56]", arg685_1: "i64[]", arg686_1: "f32[56]", arg687_1: "f32[56]", arg688_1: "i64[]", arg689_1: "f32[56]", arg690_1: "f32[56]", arg691_1: "i64[]", arg692_1: "f32[56]", arg693_1: "f32[56]", arg694_1: "i64[]", arg695_1: "f32[56]", arg696_1: "f32[56]", arg697_1: "i64[]", arg698_1: "f32[56]", arg699_1: "f32[56]", arg700_1: "i64[]", arg701_1: "f32[1024]", arg702_1: "f32[1024]", arg703_1: "i64[]", arg704_1: "f32[448]", arg705_1: "f32[448]", arg706_1: "i64[]", arg707_1: "f32[56]", arg708_1: "f32[56]", arg709_1: "i64[]", arg710_1: "f32[56]", arg711_1: "f32[56]", arg712_1: "i64[]", arg713_1: "f32[56]", arg714_1: "f32[56]", arg715_1: "i64[]", arg716_1: "f32[56]", arg717_1: "f32[56]", arg718_1: "i64[]", arg719_1: "f32[56]", arg720_1: "f32[56]", arg721_1: "i64[]", arg722_1: "f32[56]", arg723_1: "f32[56]", arg724_1: "i64[]", arg725_1: "f32[56]", arg726_1: "f32[56]", arg727_1: "i64[]", arg728_1: "f32[1024]", arg729_1: "f32[1024]", arg730_1: "i64[]", arg731_1: "f32[448]", arg732_1: "f32[448]", arg733_1: "i64[]", arg734_1: "f32[56]", arg735_1: "f32[56]", arg736_1: "i64[]", arg737_1: "f32[56]", arg738_1: "f32[56]", arg739_1: "i64[]", arg740_1: "f32[56]", arg741_1: "f32[56]", arg742_1: "i64[]", arg743_1: "f32[56]", arg744_1: "f32[56]", arg745_1: "i64[]", arg746_1: "f32[56]", arg747_1: "f32[56]", arg748_1: "i64[]", arg749_1: "f32[56]", arg750_1: "f32[56]", arg751_1: "i64[]", arg752_1: "f32[56]", arg753_1: "f32[56]", arg754_1: "i64[]", arg755_1: "f32[1024]", arg756_1: "f32[1024]", arg757_1: "i64[]", arg758_1: "f32[448]", arg759_1: "f32[448]", arg760_1: "i64[]", arg761_1: "f32[56]", arg762_1: "f32[56]", arg763_1: "i64[]", arg764_1: "f32[56]", arg765_1: "f32[56]", arg766_1: "i64[]", arg767_1: "f32[56]", arg768_1: "f32[56]", arg769_1: "i64[]", arg770_1: "f32[56]", arg771_1: "f32[56]", arg772_1: "i64[]", arg773_1: "f32[56]", arg774_1: "f32[56]", arg775_1: "i64[]", arg776_1: "f32[56]", arg777_1: "f32[56]", arg778_1: "i64[]", arg779_1: "f32[56]", arg780_1: "f32[56]", arg781_1: "i64[]", arg782_1: "f32[1024]", arg783_1: "f32[1024]", arg784_1: "i64[]", arg785_1: "f32[448]", arg786_1: "f32[448]", arg787_1: "i64[]", arg788_1: "f32[56]", arg789_1: "f32[56]", arg790_1: "i64[]", arg791_1: "f32[56]", arg792_1: "f32[56]", arg793_1: "i64[]", arg794_1: "f32[56]", arg795_1: "f32[56]", arg796_1: "i64[]", arg797_1: "f32[56]", arg798_1: "f32[56]", arg799_1: "i64[]", arg800_1: "f32[56]", arg801_1: "f32[56]", arg802_1: "i64[]", arg803_1: "f32[56]", arg804_1: "f32[56]", arg805_1: "i64[]", arg806_1: "f32[56]", arg807_1: "f32[56]", arg808_1: "i64[]", arg809_1: "f32[1024]", arg810_1: "f32[1024]", arg811_1: "i64[]", arg812_1: "f32[896]", arg813_1: "f32[896]", arg814_1: "i64[]", arg815_1: "f32[112]", arg816_1: "f32[112]", arg817_1: "i64[]", arg818_1: "f32[112]", arg819_1: "f32[112]", arg820_1: "i64[]", arg821_1: "f32[112]", arg822_1: "f32[112]", arg823_1: "i64[]", arg824_1: "f32[112]", arg825_1: "f32[112]", arg826_1: "i64[]", arg827_1: "f32[112]", arg828_1: "f32[112]", arg829_1: "i64[]", arg830_1: "f32[112]", arg831_1: "f32[112]", arg832_1: "i64[]", arg833_1: "f32[112]", arg834_1: "f32[112]", arg835_1: "i64[]", arg836_1: "f32[2048]", arg837_1: "f32[2048]", arg838_1: "i64[]", arg839_1: "f32[2048]", arg840_1: "f32[2048]", arg841_1: "i64[]", arg842_1: "f32[896]", arg843_1: "f32[896]", arg844_1: "i64[]", arg845_1: "f32[112]", arg846_1: "f32[112]", arg847_1: "i64[]", arg848_1: "f32[112]", arg849_1: "f32[112]", arg850_1: "i64[]", arg851_1: "f32[112]", arg852_1: "f32[112]", arg853_1: "i64[]", arg854_1: "f32[112]", arg855_1: "f32[112]", arg856_1: "i64[]", arg857_1: "f32[112]", arg858_1: "f32[112]", arg859_1: "i64[]", arg860_1: "f32[112]", arg861_1: "f32[112]", arg862_1: "i64[]", arg863_1: "f32[112]", arg864_1: "f32[112]", arg865_1: "i64[]", arg866_1: "f32[2048]", arg867_1: "f32[2048]", arg868_1: "i64[]", arg869_1: "f32[896]", arg870_1: "f32[896]", arg871_1: "i64[]", arg872_1: "f32[112]", arg873_1: "f32[112]", arg874_1: "i64[]", arg875_1: "f32[112]", arg876_1: "f32[112]", arg877_1: "i64[]", arg878_1: "f32[112]", arg879_1: "f32[112]", arg880_1: "i64[]", arg881_1: "f32[112]", arg882_1: "f32[112]", arg883_1: "i64[]", arg884_1: "f32[112]", arg885_1: "f32[112]", arg886_1: "i64[]", arg887_1: "f32[112]", arg888_1: "f32[112]", arg889_1: "i64[]", arg890_1: "f32[112]", arg891_1: "f32[112]", arg892_1: "i64[]", arg893_1: "f32[2048]", arg894_1: "f32[2048]", arg895_1: "i64[]", arg896_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg896_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg896_1 = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[64]" = torch.ops.aten.add.Tensor(arg450_1, 1e-05);  arg450_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
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
    convolution_1: "f32[8, 112, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg3_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_8: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
    unsqueeze_9: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    add_2: "f32[112]" = torch.ops.aten.add.Tensor(arg453_1, 1e-05);  arg453_1 = None
    sqrt_1: "f32[112]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_4: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_13: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_15: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 112, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_1: "f32[8, 112, 56, 56]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_10: "f32[8, 14, 56, 56]" = split_with_sizes_1[0];  split_with_sizes_1 = None
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_19: "f32[8, 14, 56, 56]" = split_with_sizes_2[1];  split_with_sizes_2 = None
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_28: "f32[8, 14, 56, 56]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_37: "f32[8, 14, 56, 56]" = split_with_sizes_4[3];  split_with_sizes_4 = None
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_46: "f32[8, 14, 56, 56]" = split_with_sizes_5[4];  split_with_sizes_5 = None
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_55: "f32[8, 14, 56, 56]" = split_with_sizes_6[5];  split_with_sizes_6 = None
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_64: "f32[8, 14, 56, 56]" = split_with_sizes_7[6];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(relu_1, [14, 14, 14, 14, 14, 14, 14, 14], 1);  relu_1 = None
    getitem_73: "f32[8, 14, 56, 56]" = split_with_sizes_8[7];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_2: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_10, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_10 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_16: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
    unsqueeze_17: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    add_4: "f32[14]" = torch.ops.aten.add.Tensor(arg456_1, 1e-05);  arg456_1 = None
    sqrt_2: "f32[14]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_7: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_21: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_23: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_2: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_3: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_19, arg9_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_19 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_24: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg458_1, -1);  arg458_1 = None
    unsqueeze_25: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    sub_3: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    add_6: "f32[14]" = torch.ops.aten.add.Tensor(arg459_1, 1e-05);  arg459_1 = None
    sqrt_3: "f32[14]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_26: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    mul_10: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_29: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_31: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_3: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_4: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_28, arg12_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_28 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_32: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg461_1, -1);  arg461_1 = None
    unsqueeze_33: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    sub_4: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    add_8: "f32[14]" = torch.ops.aten.add.Tensor(arg462_1, 1e-05);  arg462_1 = None
    sqrt_4: "f32[14]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_34: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    mul_13: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_37: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_39: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_4: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_5: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_37, arg15_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_37 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_40: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
    unsqueeze_41: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    sub_5: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    add_10: "f32[14]" = torch.ops.aten.add.Tensor(arg465_1, 1e-05);  arg465_1 = None
    sqrt_5: "f32[14]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_42: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    mul_16: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_45: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_47: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_5: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_6: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_46, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_46 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_48: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
    unsqueeze_49: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    sub_6: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    add_12: "f32[14]" = torch.ops.aten.add.Tensor(arg468_1, 1e-05);  arg468_1 = None
    sqrt_6: "f32[14]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_50: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    mul_19: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_53: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_55: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_6: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_7: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_55, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_55 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_56: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
    unsqueeze_57: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    sub_7: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    add_14: "f32[14]" = torch.ops.aten.add.Tensor(arg471_1, 1e-05);  arg471_1 = None
    sqrt_7: "f32[14]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_58: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    mul_22: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_61: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_63: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_7: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_8: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_64, arg24_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_64 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_64: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
    unsqueeze_65: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    sub_8: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    add_16: "f32[14]" = torch.ops.aten.add.Tensor(arg474_1, 1e-05);  arg474_1 = None
    sqrt_8: "f32[14]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_66: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    mul_25: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_69: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_71: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_8: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d: "f32[8, 14, 56, 56]" = torch.ops.aten.avg_pool2d.default(getitem_73, [3, 3], [1, 1], [1, 1]);  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([relu_2, relu_3, relu_4, relu_5, relu_6, relu_7, relu_8, avg_pool2d], 1);  relu_2 = relu_3 = relu_4 = relu_5 = relu_6 = relu_7 = relu_8 = avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_9: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat, arg27_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_72: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
    unsqueeze_73: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    sub_9: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    add_18: "f32[256]" = torch.ops.aten.add.Tensor(arg477_1, 1e-05);  arg477_1 = None
    sqrt_9: "f32[256]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_9: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_74: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    mul_28: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_77: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_79: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_19: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_10: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg30_1 = None
    unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
    unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    sub_10: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    add_20: "f32[256]" = torch.ops.aten.add.Tensor(arg480_1, 1e-05);  arg480_1 = None
    sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    mul_31: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_21: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_22: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_21);  add_19 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_9: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_11: "f32[8, 112, 56, 56]" = torch.ops.aten.convolution.default(relu_9, arg33_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_88: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
    unsqueeze_89: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    sub_11: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    add_23: "f32[112]" = torch.ops.aten.add.Tensor(arg483_1, 1e-05);  arg483_1 = None
    sqrt_11: "f32[112]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_90: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    mul_34: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_93: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_95: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[8, 112, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_10: "f32[8, 112, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_82: "f32[8, 14, 56, 56]" = split_with_sizes_10[0];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_91: "f32[8, 14, 56, 56]" = split_with_sizes_11[1];  split_with_sizes_11 = None
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_100: "f32[8, 14, 56, 56]" = split_with_sizes_12[2];  split_with_sizes_12 = None
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_109: "f32[8, 14, 56, 56]" = split_with_sizes_13[3];  split_with_sizes_13 = None
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_118: "f32[8, 14, 56, 56]" = split_with_sizes_14[4];  split_with_sizes_14 = None
    split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_127: "f32[8, 14, 56, 56]" = split_with_sizes_15[5];  split_with_sizes_15 = None
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_136: "f32[8, 14, 56, 56]" = split_with_sizes_16[6];  split_with_sizes_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(relu_10, [14, 14, 14, 14, 14, 14, 14, 14], 1);  relu_10 = None
    getitem_145: "f32[8, 14, 56, 56]" = split_with_sizes_17[7];  split_with_sizes_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_12: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_82, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_82 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_96: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
    unsqueeze_97: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    sub_12: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    add_25: "f32[14]" = torch.ops.aten.add.Tensor(arg486_1, 1e-05);  arg486_1 = None
    sqrt_12: "f32[14]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_12: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_98: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    mul_37: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_101: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_103: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_26: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_11: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_27: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_11, getitem_91);  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_13: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_27, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_27 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_104: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
    unsqueeze_105: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    sub_13: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    add_28: "f32[14]" = torch.ops.aten.add.Tensor(arg489_1, 1e-05);  arg489_1 = None
    sqrt_13: "f32[14]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_106: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    mul_40: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_109: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_111: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_12: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_30: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_12, getitem_100);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_14: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_30, arg42_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_30 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_112: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg491_1, -1);  arg491_1 = None
    unsqueeze_113: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    sub_14: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    add_31: "f32[14]" = torch.ops.aten.add.Tensor(arg492_1, 1e-05);  arg492_1 = None
    sqrt_14: "f32[14]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_14: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_114: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    mul_43: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_117: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_119: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_32: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_13: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_33: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_13, getitem_109);  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_15: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_33, arg45_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_33 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_120: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
    unsqueeze_121: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    sub_15: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    add_34: "f32[14]" = torch.ops.aten.add.Tensor(arg495_1, 1e-05);  arg495_1 = None
    sqrt_15: "f32[14]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_15: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_122: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    mul_46: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_125: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_127: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_35: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_14: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_36: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_14, getitem_118);  getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_16: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_36, arg48_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_36 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_128: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
    unsqueeze_129: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    sub_16: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    add_37: "f32[14]" = torch.ops.aten.add.Tensor(arg498_1, 1e-05);  arg498_1 = None
    sqrt_16: "f32[14]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    reciprocal_16: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_130: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    mul_49: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_133: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_135: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_38: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_15: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_38);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_39: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_15, getitem_127);  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_17: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_39, arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_39 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_136: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
    unsqueeze_137: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    sub_17: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    add_40: "f32[14]" = torch.ops.aten.add.Tensor(arg501_1, 1e-05);  arg501_1 = None
    sqrt_17: "f32[14]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_17: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_138: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    mul_52: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_141: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_143: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_41: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_16: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_42: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_16, getitem_136);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_18: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_42, arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_42 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_144: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
    unsqueeze_145: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    sub_18: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    add_43: "f32[14]" = torch.ops.aten.add.Tensor(arg504_1, 1e-05);  arg504_1 = None
    sqrt_18: "f32[14]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_18: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_146: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    mul_55: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_149: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_151: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_44: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_17: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_1: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([relu_11, relu_12, relu_13, relu_14, relu_15, relu_16, relu_17, getitem_145], 1);  relu_11 = relu_12 = relu_13 = relu_14 = relu_15 = relu_16 = relu_17 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_19: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_1, arg57_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_1 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_152: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
    unsqueeze_153: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    sub_19: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
    add_45: "f32[256]" = torch.ops.aten.add.Tensor(arg507_1, 1e-05);  arg507_1 = None
    sqrt_19: "f32[256]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_19: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_154: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    mul_58: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_157: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_159: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_46: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_47: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_46, relu_9);  add_46 = relu_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_18: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_20: "f32[8, 112, 56, 56]" = torch.ops.aten.convolution.default(relu_18, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_160: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
    unsqueeze_161: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    sub_20: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
    add_48: "f32[112]" = torch.ops.aten.add.Tensor(arg510_1, 1e-05);  arg510_1 = None
    sqrt_20: "f32[112]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_20: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_162: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    mul_61: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_165: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_167: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_49: "f32[8, 112, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_19: "f32[8, 112, 56, 56]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_154: "f32[8, 14, 56, 56]" = split_with_sizes_19[0];  split_with_sizes_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_20 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_163: "f32[8, 14, 56, 56]" = split_with_sizes_20[1];  split_with_sizes_20 = None
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_172: "f32[8, 14, 56, 56]" = split_with_sizes_21[2];  split_with_sizes_21 = None
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_181: "f32[8, 14, 56, 56]" = split_with_sizes_22[3];  split_with_sizes_22 = None
    split_with_sizes_23 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_190: "f32[8, 14, 56, 56]" = split_with_sizes_23[4];  split_with_sizes_23 = None
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_199: "f32[8, 14, 56, 56]" = split_with_sizes_24[5];  split_with_sizes_24 = None
    split_with_sizes_25 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1)
    getitem_208: "f32[8, 14, 56, 56]" = split_with_sizes_25[6];  split_with_sizes_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(relu_19, [14, 14, 14, 14, 14, 14, 14, 14], 1);  relu_19 = None
    getitem_217: "f32[8, 14, 56, 56]" = split_with_sizes_26[7];  split_with_sizes_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_21: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_154, arg63_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_154 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_168: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg512_1, -1);  arg512_1 = None
    unsqueeze_169: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    sub_21: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
    add_50: "f32[14]" = torch.ops.aten.add.Tensor(arg513_1, 1e-05);  arg513_1 = None
    sqrt_21: "f32[14]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_21: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_170: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    mul_64: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_173: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_175: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_51: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_20: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_52: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_20, getitem_163);  getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_22: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_52, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_52 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_176: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg515_1, -1);  arg515_1 = None
    unsqueeze_177: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    sub_22: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
    add_53: "f32[14]" = torch.ops.aten.add.Tensor(arg516_1, 1e-05);  arg516_1 = None
    sqrt_22: "f32[14]" = torch.ops.aten.sqrt.default(add_53);  add_53 = None
    reciprocal_22: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_178: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    mul_67: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_181: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_183: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_54: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_21: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_54);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_55: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_21, getitem_172);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_23: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_55, arg69_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_55 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_184: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg518_1, -1);  arg518_1 = None
    unsqueeze_185: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    sub_23: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
    add_56: "f32[14]" = torch.ops.aten.add.Tensor(arg519_1, 1e-05);  arg519_1 = None
    sqrt_23: "f32[14]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_23: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_186: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    mul_70: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_189: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_191: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_57: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_22: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_58: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_22, getitem_181);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_24: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_58, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_58 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_192: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg521_1, -1);  arg521_1 = None
    unsqueeze_193: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    sub_24: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
    add_59: "f32[14]" = torch.ops.aten.add.Tensor(arg522_1, 1e-05);  arg522_1 = None
    sqrt_24: "f32[14]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_24: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_194: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    mul_73: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_197: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_199: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_60: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_23: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_60);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_61: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_23, getitem_190);  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_25: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_61, arg75_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_61 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_200: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
    unsqueeze_201: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    sub_25: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
    add_62: "f32[14]" = torch.ops.aten.add.Tensor(arg525_1, 1e-05);  arg525_1 = None
    sqrt_25: "f32[14]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_25: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_202: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    mul_76: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_205: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_207: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_63: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_24: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_64: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_24, getitem_199);  getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_26: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_64, arg78_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_64 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_208: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg527_1, -1);  arg527_1 = None
    unsqueeze_209: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    sub_26: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
    add_65: "f32[14]" = torch.ops.aten.add.Tensor(arg528_1, 1e-05);  arg528_1 = None
    sqrt_26: "f32[14]" = torch.ops.aten.sqrt.default(add_65);  add_65 = None
    reciprocal_26: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_210: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    mul_79: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_213: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_215: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_66: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_25: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_66);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_67: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_25, getitem_208);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_27: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_67, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_67 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_216: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg530_1, -1);  arg530_1 = None
    unsqueeze_217: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    sub_27: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  convolution_27 = unsqueeze_217 = None
    add_68: "f32[14]" = torch.ops.aten.add.Tensor(arg531_1, 1e-05);  arg531_1 = None
    sqrt_27: "f32[14]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_27: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_218: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    mul_82: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_221: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_223: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_69: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_26: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_2: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([relu_20, relu_21, relu_22, relu_23, relu_24, relu_25, relu_26, getitem_217], 1);  relu_20 = relu_21 = relu_22 = relu_23 = relu_24 = relu_25 = relu_26 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_28: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_2, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_224: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg533_1, -1);  arg533_1 = None
    unsqueeze_225: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    sub_28: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  convolution_28 = unsqueeze_225 = None
    add_70: "f32[256]" = torch.ops.aten.add.Tensor(arg534_1, 1e-05);  arg534_1 = None
    sqrt_28: "f32[256]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_28: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_226: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    mul_85: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_229: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_231: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_71: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_72: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_71, relu_18);  add_71 = relu_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_27: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_29: "f32[8, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_27, arg87_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_232: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg536_1, -1);  arg536_1 = None
    unsqueeze_233: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    sub_29: "f32[8, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  convolution_29 = unsqueeze_233 = None
    add_73: "f32[224]" = torch.ops.aten.add.Tensor(arg537_1, 1e-05);  arg537_1 = None
    sqrt_29: "f32[224]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_29: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_234: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    mul_88: "f32[8, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_237: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[8, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_239: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_74: "f32[8, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_28: "f32[8, 224, 56, 56]" = torch.ops.aten.relu.default(add_74);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_28 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_226: "f32[8, 28, 56, 56]" = split_with_sizes_28[0];  split_with_sizes_28 = None
    split_with_sizes_29 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_235: "f32[8, 28, 56, 56]" = split_with_sizes_29[1];  split_with_sizes_29 = None
    split_with_sizes_30 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_244: "f32[8, 28, 56, 56]" = split_with_sizes_30[2];  split_with_sizes_30 = None
    split_with_sizes_31 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_253: "f32[8, 28, 56, 56]" = split_with_sizes_31[3];  split_with_sizes_31 = None
    split_with_sizes_32 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_262: "f32[8, 28, 56, 56]" = split_with_sizes_32[4];  split_with_sizes_32 = None
    split_with_sizes_33 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_271: "f32[8, 28, 56, 56]" = split_with_sizes_33[5];  split_with_sizes_33 = None
    split_with_sizes_34 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_280: "f32[8, 28, 56, 56]" = split_with_sizes_34[6];  split_with_sizes_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_35 = torch.ops.aten.split_with_sizes.default(relu_28, [28, 28, 28, 28, 28, 28, 28, 28], 1);  relu_28 = None
    getitem_289: "f32[8, 28, 56, 56]" = split_with_sizes_35[7];  split_with_sizes_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_30: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_226, arg90_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_226 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_240: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg539_1, -1);  arg539_1 = None
    unsqueeze_241: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    sub_30: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  convolution_30 = unsqueeze_241 = None
    add_75: "f32[28]" = torch.ops.aten.add.Tensor(arg540_1, 1e-05);  arg540_1 = None
    sqrt_30: "f32[28]" = torch.ops.aten.sqrt.default(add_75);  add_75 = None
    reciprocal_30: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_242: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    mul_91: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_245: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_247: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_76: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_29: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_76);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_31: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_235, arg93_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_235 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_248: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg542_1, -1);  arg542_1 = None
    unsqueeze_249: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    sub_31: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
    add_77: "f32[28]" = torch.ops.aten.add.Tensor(arg543_1, 1e-05);  arg543_1 = None
    sqrt_31: "f32[28]" = torch.ops.aten.sqrt.default(add_77);  add_77 = None
    reciprocal_31: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_250: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    mul_94: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_253: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_255: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_78: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_30: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_78);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_32: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_244, arg96_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_244 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_256: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
    unsqueeze_257: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    sub_32: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
    add_79: "f32[28]" = torch.ops.aten.add.Tensor(arg546_1, 1e-05);  arg546_1 = None
    sqrt_32: "f32[28]" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
    reciprocal_32: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_258: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    mul_97: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_261: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_263: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_80: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_31: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_80);  add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_33: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_253, arg99_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_253 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_264: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg548_1, -1);  arg548_1 = None
    unsqueeze_265: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    sub_33: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
    add_81: "f32[28]" = torch.ops.aten.add.Tensor(arg549_1, 1e-05);  arg549_1 = None
    sqrt_33: "f32[28]" = torch.ops.aten.sqrt.default(add_81);  add_81 = None
    reciprocal_33: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_266: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    mul_100: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_269: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_271: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_82: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_32: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_82);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_34: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_262, arg102_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_262 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_272: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg551_1, -1);  arg551_1 = None
    unsqueeze_273: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    sub_34: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
    add_83: "f32[28]" = torch.ops.aten.add.Tensor(arg552_1, 1e-05);  arg552_1 = None
    sqrt_34: "f32[28]" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
    reciprocal_34: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_274: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    mul_103: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_277: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_279: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_84: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_33: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_35: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_271, arg105_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_271 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_280: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
    unsqueeze_281: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    sub_35: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
    add_85: "f32[28]" = torch.ops.aten.add.Tensor(arg555_1, 1e-05);  arg555_1 = None
    sqrt_35: "f32[28]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_35: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_282: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    mul_106: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_285: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_287: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_86: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_34: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_36: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_280, arg108_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_280 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_288: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg557_1, -1);  arg557_1 = None
    unsqueeze_289: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    sub_36: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
    add_87: "f32[28]" = torch.ops.aten.add.Tensor(arg558_1, 1e-05);  arg558_1 = None
    sqrt_36: "f32[28]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_36: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_290: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    mul_109: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_293: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_295: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_88: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_35: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_1: "f32[8, 28, 28, 28]" = torch.ops.aten.avg_pool2d.default(getitem_289, [3, 3], [2, 2], [1, 1]);  getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_3: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_29, relu_30, relu_31, relu_32, relu_33, relu_34, relu_35, avg_pool2d_1], 1);  relu_29 = relu_30 = relu_31 = relu_32 = relu_33 = relu_34 = relu_35 = avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_37: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_3, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_3 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_296: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg560_1, -1);  arg560_1 = None
    unsqueeze_297: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    sub_37: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
    add_89: "f32[512]" = torch.ops.aten.add.Tensor(arg561_1, 1e-05);  arg561_1 = None
    sqrt_37: "f32[512]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_37: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_298: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    mul_112: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_301: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_303: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_90: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_38: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_27, arg114_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_27 = arg114_1 = None
    unsqueeze_304: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg563_1, -1);  arg563_1 = None
    unsqueeze_305: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    sub_38: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
    add_91: "f32[512]" = torch.ops.aten.add.Tensor(arg564_1, 1e-05);  arg564_1 = None
    sqrt_38: "f32[512]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    reciprocal_38: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_306: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    mul_115: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_309: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_311: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_92: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_93: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_90, add_92);  add_90 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_36: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_39: "f32[8, 224, 28, 28]" = torch.ops.aten.convolution.default(relu_36, arg117_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_312: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg566_1, -1);  arg566_1 = None
    unsqueeze_313: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    sub_39: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
    add_94: "f32[224]" = torch.ops.aten.add.Tensor(arg567_1, 1e-05);  arg567_1 = None
    sqrt_39: "f32[224]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_39: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_314: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    mul_118: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_317: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_319: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_95: "f32[8, 224, 28, 28]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_37: "f32[8, 224, 28, 28]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_37 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_298: "f32[8, 28, 28, 28]" = split_with_sizes_37[0];  split_with_sizes_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_38 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_307: "f32[8, 28, 28, 28]" = split_with_sizes_38[1];  split_with_sizes_38 = None
    split_with_sizes_39 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_316: "f32[8, 28, 28, 28]" = split_with_sizes_39[2];  split_with_sizes_39 = None
    split_with_sizes_40 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_325: "f32[8, 28, 28, 28]" = split_with_sizes_40[3];  split_with_sizes_40 = None
    split_with_sizes_41 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_334: "f32[8, 28, 28, 28]" = split_with_sizes_41[4];  split_with_sizes_41 = None
    split_with_sizes_42 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_343: "f32[8, 28, 28, 28]" = split_with_sizes_42[5];  split_with_sizes_42 = None
    split_with_sizes_43 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_352: "f32[8, 28, 28, 28]" = split_with_sizes_43[6];  split_with_sizes_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_44 = torch.ops.aten.split_with_sizes.default(relu_37, [28, 28, 28, 28, 28, 28, 28, 28], 1);  relu_37 = None
    getitem_361: "f32[8, 28, 28, 28]" = split_with_sizes_44[7];  split_with_sizes_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_40: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_298, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_298 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_320: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg569_1, -1);  arg569_1 = None
    unsqueeze_321: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    sub_40: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
    add_96: "f32[28]" = torch.ops.aten.add.Tensor(arg570_1, 1e-05);  arg570_1 = None
    sqrt_40: "f32[28]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_40: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_322: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    mul_121: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_325: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_327: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_97: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_38: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_98: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_38, getitem_307);  getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_41: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_98, arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_98 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_328: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg572_1, -1);  arg572_1 = None
    unsqueeze_329: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    sub_41: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
    add_99: "f32[28]" = torch.ops.aten.add.Tensor(arg573_1, 1e-05);  arg573_1 = None
    sqrt_41: "f32[28]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_41: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_330: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    mul_124: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_333: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_335: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_100: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_39: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_101: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_39, getitem_316);  getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_42: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_101, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_101 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_336: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg575_1, -1);  arg575_1 = None
    unsqueeze_337: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    sub_42: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
    add_102: "f32[28]" = torch.ops.aten.add.Tensor(arg576_1, 1e-05);  arg576_1 = None
    sqrt_42: "f32[28]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
    reciprocal_42: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_338: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    mul_127: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_341: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_343: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_103: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_40: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_103);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_104: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_40, getitem_325);  getitem_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_43: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_104, arg129_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_104 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_344: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg578_1, -1);  arg578_1 = None
    unsqueeze_345: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    sub_43: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
    add_105: "f32[28]" = torch.ops.aten.add.Tensor(arg579_1, 1e-05);  arg579_1 = None
    sqrt_43: "f32[28]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_43: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_346: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    mul_130: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_349: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_351: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_106: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_41: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_106);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_107: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_41, getitem_334);  getitem_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_44: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_107, arg132_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_107 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_352: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg581_1, -1);  arg581_1 = None
    unsqueeze_353: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    sub_44: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
    add_108: "f32[28]" = torch.ops.aten.add.Tensor(arg582_1, 1e-05);  arg582_1 = None
    sqrt_44: "f32[28]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_44: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_354: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    mul_133: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_357: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
    unsqueeze_359: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_109: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_42: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_110: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_42, getitem_343);  getitem_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_45: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_110, arg135_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_110 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_360: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
    unsqueeze_361: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    sub_45: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
    add_111: "f32[28]" = torch.ops.aten.add.Tensor(arg585_1, 1e-05);  arg585_1 = None
    sqrt_45: "f32[28]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
    reciprocal_45: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_362: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    mul_136: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
    unsqueeze_365: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_367: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_112: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_43: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_112);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_113: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_43, getitem_352);  getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_46: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_113, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_113 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_368: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg587_1, -1);  arg587_1 = None
    unsqueeze_369: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    sub_46: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
    add_114: "f32[28]" = torch.ops.aten.add.Tensor(arg588_1, 1e-05);  arg588_1 = None
    sqrt_46: "f32[28]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
    reciprocal_46: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_370: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    mul_139: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_373: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
    unsqueeze_375: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_115: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_44: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_115);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_4: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_38, relu_39, relu_40, relu_41, relu_42, relu_43, relu_44, getitem_361], 1);  relu_38 = relu_39 = relu_40 = relu_41 = relu_42 = relu_43 = relu_44 = getitem_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_47: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_4, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_376: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg590_1, -1);  arg590_1 = None
    unsqueeze_377: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    sub_47: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
    add_116: "f32[512]" = torch.ops.aten.add.Tensor(arg591_1, 1e-05);  arg591_1 = None
    sqrt_47: "f32[512]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
    reciprocal_47: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_378: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    mul_142: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_381: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_383: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_117: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_118: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_117, relu_36);  add_117 = relu_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_45: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_118);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_48: "f32[8, 224, 28, 28]" = torch.ops.aten.convolution.default(relu_45, arg144_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_384: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg593_1, -1);  arg593_1 = None
    unsqueeze_385: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    sub_48: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
    add_119: "f32[224]" = torch.ops.aten.add.Tensor(arg594_1, 1e-05);  arg594_1 = None
    sqrt_48: "f32[224]" = torch.ops.aten.sqrt.default(add_119);  add_119 = None
    reciprocal_48: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_386: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    mul_145: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
    unsqueeze_389: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
    unsqueeze_391: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_120: "f32[8, 224, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_46: "f32[8, 224, 28, 28]" = torch.ops.aten.relu.default(add_120);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_46 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_370: "f32[8, 28, 28, 28]" = split_with_sizes_46[0];  split_with_sizes_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_47 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_379: "f32[8, 28, 28, 28]" = split_with_sizes_47[1];  split_with_sizes_47 = None
    split_with_sizes_48 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_388: "f32[8, 28, 28, 28]" = split_with_sizes_48[2];  split_with_sizes_48 = None
    split_with_sizes_49 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_397: "f32[8, 28, 28, 28]" = split_with_sizes_49[3];  split_with_sizes_49 = None
    split_with_sizes_50 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_406: "f32[8, 28, 28, 28]" = split_with_sizes_50[4];  split_with_sizes_50 = None
    split_with_sizes_51 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_415: "f32[8, 28, 28, 28]" = split_with_sizes_51[5];  split_with_sizes_51 = None
    split_with_sizes_52 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_424: "f32[8, 28, 28, 28]" = split_with_sizes_52[6];  split_with_sizes_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_53 = torch.ops.aten.split_with_sizes.default(relu_46, [28, 28, 28, 28, 28, 28, 28, 28], 1);  relu_46 = None
    getitem_433: "f32[8, 28, 28, 28]" = split_with_sizes_53[7];  split_with_sizes_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_49: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_370, arg147_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_370 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_392: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg596_1, -1);  arg596_1 = None
    unsqueeze_393: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    sub_49: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
    add_121: "f32[28]" = torch.ops.aten.add.Tensor(arg597_1, 1e-05);  arg597_1 = None
    sqrt_49: "f32[28]" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
    reciprocal_49: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_394: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    mul_148: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
    unsqueeze_397: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
    unsqueeze_399: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_122: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_47: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_122);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_123: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_47, getitem_379);  getitem_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_50: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_123, arg150_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_123 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_400: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg599_1, -1);  arg599_1 = None
    unsqueeze_401: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    sub_50: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
    add_124: "f32[28]" = torch.ops.aten.add.Tensor(arg600_1, 1e-05);  arg600_1 = None
    sqrt_50: "f32[28]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_50: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_402: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    mul_151: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
    unsqueeze_405: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
    unsqueeze_407: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_125: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_48: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_125);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_126: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_48, getitem_388);  getitem_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_51: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_126, arg153_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_126 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_408: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg602_1, -1);  arg602_1 = None
    unsqueeze_409: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    sub_51: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
    add_127: "f32[28]" = torch.ops.aten.add.Tensor(arg603_1, 1e-05);  arg603_1 = None
    sqrt_51: "f32[28]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_51: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_410: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    mul_154: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
    unsqueeze_413: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
    unsqueeze_415: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_128: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_49: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_129: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_49, getitem_397);  getitem_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_52: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_129, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_129 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_416: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
    unsqueeze_417: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    sub_52: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
    add_130: "f32[28]" = torch.ops.aten.add.Tensor(arg606_1, 1e-05);  arg606_1 = None
    sqrt_52: "f32[28]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
    reciprocal_52: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_418: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    mul_157: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
    unsqueeze_421: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
    unsqueeze_423: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_131: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_50: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_131);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_132: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_50, getitem_406);  getitem_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_53: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_132, arg159_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_132 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_424: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg608_1, -1);  arg608_1 = None
    unsqueeze_425: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    sub_53: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
    add_133: "f32[28]" = torch.ops.aten.add.Tensor(arg609_1, 1e-05);  arg609_1 = None
    sqrt_53: "f32[28]" = torch.ops.aten.sqrt.default(add_133);  add_133 = None
    reciprocal_53: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_159: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_426: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_427: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    mul_160: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
    unsqueeze_429: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_161: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
    unsqueeze_430: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
    unsqueeze_431: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_134: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_51: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_134);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_135: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_51, getitem_415);  getitem_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_54: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_135, arg162_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_135 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_432: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg611_1, -1);  arg611_1 = None
    unsqueeze_433: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    sub_54: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
    add_136: "f32[28]" = torch.ops.aten.add.Tensor(arg612_1, 1e-05);  arg612_1 = None
    sqrt_54: "f32[28]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
    reciprocal_54: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_162: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_434: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_435: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    mul_163: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
    unsqueeze_437: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_164: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
    unsqueeze_438: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
    unsqueeze_439: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_137: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_52: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_137);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_138: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_52, getitem_424);  getitem_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_55: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_138, arg165_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_138 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_440: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg614_1, -1);  arg614_1 = None
    unsqueeze_441: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    sub_55: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
    add_139: "f32[28]" = torch.ops.aten.add.Tensor(arg615_1, 1e-05);  arg615_1 = None
    sqrt_55: "f32[28]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
    reciprocal_55: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_165: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_442: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_443: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    mul_166: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
    unsqueeze_445: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_167: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
    unsqueeze_446: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
    unsqueeze_447: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_140: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_53: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_140);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_5: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_47, relu_48, relu_49, relu_50, relu_51, relu_52, relu_53, getitem_433], 1);  relu_47 = relu_48 = relu_49 = relu_50 = relu_51 = relu_52 = relu_53 = getitem_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_56: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_5, arg168_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_448: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg617_1, -1);  arg617_1 = None
    unsqueeze_449: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    sub_56: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_449);  convolution_56 = unsqueeze_449 = None
    add_141: "f32[512]" = torch.ops.aten.add.Tensor(arg618_1, 1e-05);  arg618_1 = None
    sqrt_56: "f32[512]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
    reciprocal_56: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_168: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_450: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_451: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    mul_169: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
    unsqueeze_453: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_170: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
    unsqueeze_454: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
    unsqueeze_455: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_142: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_143: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_142, relu_45);  add_142 = relu_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_54: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_143);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_57: "f32[8, 224, 28, 28]" = torch.ops.aten.convolution.default(relu_54, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_456: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg620_1, -1);  arg620_1 = None
    unsqueeze_457: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    sub_57: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
    add_144: "f32[224]" = torch.ops.aten.add.Tensor(arg621_1, 1e-05);  arg621_1 = None
    sqrt_57: "f32[224]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
    reciprocal_57: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_171: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_458: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
    unsqueeze_459: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    mul_172: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
    unsqueeze_461: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_173: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
    unsqueeze_462: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
    unsqueeze_463: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_145: "f32[8, 224, 28, 28]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_55: "f32[8, 224, 28, 28]" = torch.ops.aten.relu.default(add_145);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_55 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_442: "f32[8, 28, 28, 28]" = split_with_sizes_55[0];  split_with_sizes_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_56 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_451: "f32[8, 28, 28, 28]" = split_with_sizes_56[1];  split_with_sizes_56 = None
    split_with_sizes_57 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_460: "f32[8, 28, 28, 28]" = split_with_sizes_57[2];  split_with_sizes_57 = None
    split_with_sizes_58 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_469: "f32[8, 28, 28, 28]" = split_with_sizes_58[3];  split_with_sizes_58 = None
    split_with_sizes_59 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_478: "f32[8, 28, 28, 28]" = split_with_sizes_59[4];  split_with_sizes_59 = None
    split_with_sizes_60 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_487: "f32[8, 28, 28, 28]" = split_with_sizes_60[5];  split_with_sizes_60 = None
    split_with_sizes_61 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1)
    getitem_496: "f32[8, 28, 28, 28]" = split_with_sizes_61[6];  split_with_sizes_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_62 = torch.ops.aten.split_with_sizes.default(relu_55, [28, 28, 28, 28, 28, 28, 28, 28], 1);  relu_55 = None
    getitem_505: "f32[8, 28, 28, 28]" = split_with_sizes_62[7];  split_with_sizes_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_58: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_442, arg174_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_442 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_464: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg623_1, -1);  arg623_1 = None
    unsqueeze_465: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    sub_58: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
    add_146: "f32[28]" = torch.ops.aten.add.Tensor(arg624_1, 1e-05);  arg624_1 = None
    sqrt_58: "f32[28]" = torch.ops.aten.sqrt.default(add_146);  add_146 = None
    reciprocal_58: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_174: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_466: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
    unsqueeze_467: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    mul_175: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
    unsqueeze_469: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_176: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
    unsqueeze_470: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
    unsqueeze_471: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_147: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_56: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_147);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_148: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_56, getitem_451);  getitem_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_59: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_148, arg177_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_148 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_472: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg626_1, -1);  arg626_1 = None
    unsqueeze_473: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    sub_59: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
    add_149: "f32[28]" = torch.ops.aten.add.Tensor(arg627_1, 1e-05);  arg627_1 = None
    sqrt_59: "f32[28]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
    reciprocal_59: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_177: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_474: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
    unsqueeze_475: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    mul_178: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
    unsqueeze_477: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_179: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
    unsqueeze_478: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
    unsqueeze_479: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_150: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_57: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_150);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_151: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_57, getitem_460);  getitem_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_60: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_151, arg180_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_151 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_480: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg629_1, -1);  arg629_1 = None
    unsqueeze_481: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    sub_60: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_481);  convolution_60 = unsqueeze_481 = None
    add_152: "f32[28]" = torch.ops.aten.add.Tensor(arg630_1, 1e-05);  arg630_1 = None
    sqrt_60: "f32[28]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
    reciprocal_60: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_180: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_482: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
    unsqueeze_483: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    mul_181: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
    unsqueeze_485: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_182: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
    unsqueeze_486: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
    unsqueeze_487: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_153: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_58: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_153);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_154: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_58, getitem_469);  getitem_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_61: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_154, arg183_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_154 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_488: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg632_1, -1);  arg632_1 = None
    unsqueeze_489: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    sub_61: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
    add_155: "f32[28]" = torch.ops.aten.add.Tensor(arg633_1, 1e-05);  arg633_1 = None
    sqrt_61: "f32[28]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
    reciprocal_61: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_183: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_490: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
    unsqueeze_491: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    mul_184: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
    unsqueeze_493: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_185: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
    unsqueeze_494: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
    unsqueeze_495: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_156: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_59: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_156);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_157: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_59, getitem_478);  getitem_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_62: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_157, arg186_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_157 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_496: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg635_1, -1);  arg635_1 = None
    unsqueeze_497: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    sub_62: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_497);  convolution_62 = unsqueeze_497 = None
    add_158: "f32[28]" = torch.ops.aten.add.Tensor(arg636_1, 1e-05);  arg636_1 = None
    sqrt_62: "f32[28]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
    reciprocal_62: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_186: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_498: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_499: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    mul_187: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
    unsqueeze_501: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_188: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
    unsqueeze_502: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
    unsqueeze_503: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_159: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_60: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_159);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_160: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_60, getitem_487);  getitem_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_63: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_160, arg189_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_160 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_504: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg638_1, -1);  arg638_1 = None
    unsqueeze_505: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    sub_63: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_505);  convolution_63 = unsqueeze_505 = None
    add_161: "f32[28]" = torch.ops.aten.add.Tensor(arg639_1, 1e-05);  arg639_1 = None
    sqrt_63: "f32[28]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
    reciprocal_63: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_189: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_506: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
    unsqueeze_507: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    mul_190: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
    unsqueeze_509: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_191: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
    unsqueeze_510: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
    unsqueeze_511: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_162: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_61: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_162);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_163: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_61, getitem_496);  getitem_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_64: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_163, arg192_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_163 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_512: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg641_1, -1);  arg641_1 = None
    unsqueeze_513: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    sub_64: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
    add_164: "f32[28]" = torch.ops.aten.add.Tensor(arg642_1, 1e-05);  arg642_1 = None
    sqrt_64: "f32[28]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
    reciprocal_64: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_192: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_514: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_515: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    mul_193: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
    unsqueeze_517: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_194: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
    unsqueeze_518: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
    unsqueeze_519: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_165: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_62: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_165);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_6: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_56, relu_57, relu_58, relu_59, relu_60, relu_61, relu_62, getitem_505], 1);  relu_56 = relu_57 = relu_58 = relu_59 = relu_60 = relu_61 = relu_62 = getitem_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_65: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_6, arg195_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_520: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg644_1, -1);  arg644_1 = None
    unsqueeze_521: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    sub_65: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
    add_166: "f32[512]" = torch.ops.aten.add.Tensor(arg645_1, 1e-05);  arg645_1 = None
    sqrt_65: "f32[512]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
    reciprocal_65: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
    mul_195: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
    unsqueeze_522: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_523: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    mul_196: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
    unsqueeze_524: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
    unsqueeze_525: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_197: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
    unsqueeze_526: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
    unsqueeze_527: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_167: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_168: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_167, relu_54);  add_167 = relu_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_63: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_168);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_66: "f32[8, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_63, arg198_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_528: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg647_1, -1);  arg647_1 = None
    unsqueeze_529: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    sub_66: "f32[8, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
    add_169: "f32[448]" = torch.ops.aten.add.Tensor(arg648_1, 1e-05);  arg648_1 = None
    sqrt_66: "f32[448]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
    reciprocal_66: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
    mul_198: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
    unsqueeze_530: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
    unsqueeze_531: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    mul_199: "f32[8, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
    unsqueeze_532: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
    unsqueeze_533: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_200: "f32[8, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
    unsqueeze_534: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
    unsqueeze_535: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_170: "f32[8, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_64: "f32[8, 448, 28, 28]" = torch.ops.aten.relu.default(add_170);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_64 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_514: "f32[8, 56, 28, 28]" = split_with_sizes_64[0];  split_with_sizes_64 = None
    split_with_sizes_65 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_523: "f32[8, 56, 28, 28]" = split_with_sizes_65[1];  split_with_sizes_65 = None
    split_with_sizes_66 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_532: "f32[8, 56, 28, 28]" = split_with_sizes_66[2];  split_with_sizes_66 = None
    split_with_sizes_67 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_541: "f32[8, 56, 28, 28]" = split_with_sizes_67[3];  split_with_sizes_67 = None
    split_with_sizes_68 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_550: "f32[8, 56, 28, 28]" = split_with_sizes_68[4];  split_with_sizes_68 = None
    split_with_sizes_69 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_559: "f32[8, 56, 28, 28]" = split_with_sizes_69[5];  split_with_sizes_69 = None
    split_with_sizes_70 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_568: "f32[8, 56, 28, 28]" = split_with_sizes_70[6];  split_with_sizes_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_71 = torch.ops.aten.split_with_sizes.default(relu_64, [56, 56, 56, 56, 56, 56, 56, 56], 1);  relu_64 = None
    getitem_577: "f32[8, 56, 28, 28]" = split_with_sizes_71[7];  split_with_sizes_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_67: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_514, arg201_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_514 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_536: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg650_1, -1);  arg650_1 = None
    unsqueeze_537: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    sub_67: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
    add_171: "f32[56]" = torch.ops.aten.add.Tensor(arg651_1, 1e-05);  arg651_1 = None
    sqrt_67: "f32[56]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
    reciprocal_67: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
    mul_201: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
    unsqueeze_538: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
    unsqueeze_539: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    mul_202: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
    unsqueeze_540: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
    unsqueeze_541: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_203: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
    unsqueeze_542: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
    unsqueeze_543: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_172: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_65: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_172);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_68: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_523, arg204_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_523 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_544: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg653_1, -1);  arg653_1 = None
    unsqueeze_545: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    sub_68: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
    add_173: "f32[56]" = torch.ops.aten.add.Tensor(arg654_1, 1e-05);  arg654_1 = None
    sqrt_68: "f32[56]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
    reciprocal_68: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
    mul_204: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
    unsqueeze_546: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
    unsqueeze_547: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    mul_205: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
    unsqueeze_548: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
    unsqueeze_549: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_206: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
    unsqueeze_550: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
    unsqueeze_551: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_174: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_66: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_174);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_69: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_532, arg207_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_532 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_552: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg656_1, -1);  arg656_1 = None
    unsqueeze_553: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    sub_69: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
    add_175: "f32[56]" = torch.ops.aten.add.Tensor(arg657_1, 1e-05);  arg657_1 = None
    sqrt_69: "f32[56]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
    reciprocal_69: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
    mul_207: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
    unsqueeze_554: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
    unsqueeze_555: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    mul_208: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
    unsqueeze_556: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
    unsqueeze_557: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_209: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
    unsqueeze_558: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
    unsqueeze_559: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_176: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_67: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_176);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_70: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_541, arg210_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_541 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_560: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg659_1, -1);  arg659_1 = None
    unsqueeze_561: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    sub_70: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
    add_177: "f32[56]" = torch.ops.aten.add.Tensor(arg660_1, 1e-05);  arg660_1 = None
    sqrt_70: "f32[56]" = torch.ops.aten.sqrt.default(add_177);  add_177 = None
    reciprocal_70: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
    mul_210: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
    unsqueeze_562: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
    unsqueeze_563: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    mul_211: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
    unsqueeze_564: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
    unsqueeze_565: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_212: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
    unsqueeze_566: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
    unsqueeze_567: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_178: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_68: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_178);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_71: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_550, arg213_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_550 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_568: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg662_1, -1);  arg662_1 = None
    unsqueeze_569: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    sub_71: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
    add_179: "f32[56]" = torch.ops.aten.add.Tensor(arg663_1, 1e-05);  arg663_1 = None
    sqrt_71: "f32[56]" = torch.ops.aten.sqrt.default(add_179);  add_179 = None
    reciprocal_71: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
    mul_213: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
    unsqueeze_570: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
    unsqueeze_571: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    mul_214: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
    unsqueeze_572: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
    unsqueeze_573: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_215: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
    unsqueeze_574: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
    unsqueeze_575: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_180: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_69: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_180);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_72: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_559, arg216_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_559 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_576: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg665_1, -1);  arg665_1 = None
    unsqueeze_577: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    sub_72: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
    add_181: "f32[56]" = torch.ops.aten.add.Tensor(arg666_1, 1e-05);  arg666_1 = None
    sqrt_72: "f32[56]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
    reciprocal_72: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
    mul_216: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
    unsqueeze_578: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
    unsqueeze_579: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    mul_217: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
    unsqueeze_580: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
    unsqueeze_581: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_218: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
    unsqueeze_582: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
    unsqueeze_583: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_182: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_70: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_182);  add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_73: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_568, arg219_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_568 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_584: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg668_1, -1);  arg668_1 = None
    unsqueeze_585: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    sub_73: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
    add_183: "f32[56]" = torch.ops.aten.add.Tensor(arg669_1, 1e-05);  arg669_1 = None
    sqrt_73: "f32[56]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
    reciprocal_73: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
    mul_219: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
    unsqueeze_586: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
    unsqueeze_587: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    mul_220: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
    unsqueeze_588: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
    unsqueeze_589: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_221: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
    unsqueeze_590: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
    unsqueeze_591: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_184: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_71: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_2: "f32[8, 56, 14, 14]" = torch.ops.aten.avg_pool2d.default(getitem_577, [3, 3], [2, 2], [1, 1]);  getitem_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_7: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_65, relu_66, relu_67, relu_68, relu_69, relu_70, relu_71, avg_pool2d_2], 1);  relu_65 = relu_66 = relu_67 = relu_68 = relu_69 = relu_70 = relu_71 = avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_74: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_7, arg222_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_592: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg671_1, -1);  arg671_1 = None
    unsqueeze_593: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    sub_74: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
    add_185: "f32[1024]" = torch.ops.aten.add.Tensor(arg672_1, 1e-05);  arg672_1 = None
    sqrt_74: "f32[1024]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
    reciprocal_74: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
    mul_222: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
    unsqueeze_594: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
    unsqueeze_595: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    mul_223: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
    unsqueeze_596: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
    unsqueeze_597: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_224: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
    unsqueeze_598: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
    unsqueeze_599: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_186: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_63, arg225_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_63 = arg225_1 = None
    unsqueeze_600: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg674_1, -1);  arg674_1 = None
    unsqueeze_601: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    sub_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
    add_187: "f32[1024]" = torch.ops.aten.add.Tensor(arg675_1, 1e-05);  arg675_1 = None
    sqrt_75: "f32[1024]" = torch.ops.aten.sqrt.default(add_187);  add_187 = None
    reciprocal_75: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
    mul_225: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
    unsqueeze_602: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
    unsqueeze_603: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    mul_226: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
    unsqueeze_604: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
    unsqueeze_605: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_227: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
    unsqueeze_606: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
    unsqueeze_607: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_188: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_189: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_186, add_188);  add_186 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_72: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_189);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_76: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_72, arg228_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_608: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg677_1, -1);  arg677_1 = None
    unsqueeze_609: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    sub_76: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
    add_190: "f32[448]" = torch.ops.aten.add.Tensor(arg678_1, 1e-05);  arg678_1 = None
    sqrt_76: "f32[448]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
    reciprocal_76: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
    mul_228: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
    unsqueeze_610: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
    unsqueeze_611: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    mul_229: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
    unsqueeze_612: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
    unsqueeze_613: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_230: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
    unsqueeze_614: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
    unsqueeze_615: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_191: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_73: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_191);  add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_73 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_586: "f32[8, 56, 14, 14]" = split_with_sizes_73[0];  split_with_sizes_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_74 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_595: "f32[8, 56, 14, 14]" = split_with_sizes_74[1];  split_with_sizes_74 = None
    split_with_sizes_75 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_604: "f32[8, 56, 14, 14]" = split_with_sizes_75[2];  split_with_sizes_75 = None
    split_with_sizes_76 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_613: "f32[8, 56, 14, 14]" = split_with_sizes_76[3];  split_with_sizes_76 = None
    split_with_sizes_77 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_622: "f32[8, 56, 14, 14]" = split_with_sizes_77[4];  split_with_sizes_77 = None
    split_with_sizes_78 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_631: "f32[8, 56, 14, 14]" = split_with_sizes_78[5];  split_with_sizes_78 = None
    split_with_sizes_79 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_640: "f32[8, 56, 14, 14]" = split_with_sizes_79[6];  split_with_sizes_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_80 = torch.ops.aten.split_with_sizes.default(relu_73, [56, 56, 56, 56, 56, 56, 56, 56], 1);  relu_73 = None
    getitem_649: "f32[8, 56, 14, 14]" = split_with_sizes_80[7];  split_with_sizes_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_77: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_586, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_586 = arg231_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_616: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg680_1, -1);  arg680_1 = None
    unsqueeze_617: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    sub_77: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
    add_192: "f32[56]" = torch.ops.aten.add.Tensor(arg681_1, 1e-05);  arg681_1 = None
    sqrt_77: "f32[56]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
    reciprocal_77: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
    mul_231: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
    unsqueeze_618: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
    unsqueeze_619: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    mul_232: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
    unsqueeze_620: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
    unsqueeze_621: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_233: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
    unsqueeze_622: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
    unsqueeze_623: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_193: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_74: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_193);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_194: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_74, getitem_595);  getitem_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_78: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_194, arg234_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_194 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_624: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg683_1, -1);  arg683_1 = None
    unsqueeze_625: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    sub_78: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
    add_195: "f32[56]" = torch.ops.aten.add.Tensor(arg684_1, 1e-05);  arg684_1 = None
    sqrt_78: "f32[56]" = torch.ops.aten.sqrt.default(add_195);  add_195 = None
    reciprocal_78: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
    mul_234: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
    unsqueeze_626: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
    unsqueeze_627: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    mul_235: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
    unsqueeze_628: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
    unsqueeze_629: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_236: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
    unsqueeze_630: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
    unsqueeze_631: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_196: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_75: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_196);  add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_197: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_75, getitem_604);  getitem_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_79: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_197, arg237_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_197 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_632: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg686_1, -1);  arg686_1 = None
    unsqueeze_633: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    sub_79: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
    add_198: "f32[56]" = torch.ops.aten.add.Tensor(arg687_1, 1e-05);  arg687_1 = None
    sqrt_79: "f32[56]" = torch.ops.aten.sqrt.default(add_198);  add_198 = None
    reciprocal_79: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
    mul_237: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
    unsqueeze_634: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
    unsqueeze_635: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    mul_238: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
    unsqueeze_636: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
    unsqueeze_637: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_239: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
    unsqueeze_638: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
    unsqueeze_639: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_199: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_76: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_199);  add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_200: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_76, getitem_613);  getitem_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_80: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_200, arg240_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_200 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_640: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
    unsqueeze_641: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
    sub_80: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
    add_201: "f32[56]" = torch.ops.aten.add.Tensor(arg690_1, 1e-05);  arg690_1 = None
    sqrt_80: "f32[56]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
    reciprocal_80: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
    mul_240: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
    unsqueeze_642: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
    unsqueeze_643: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
    mul_241: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
    unsqueeze_644: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
    unsqueeze_645: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
    mul_242: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
    unsqueeze_646: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
    unsqueeze_647: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    add_202: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_77: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_202);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_203: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_77, getitem_622);  getitem_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_81: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_203, arg243_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_203 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_648: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg692_1, -1);  arg692_1 = None
    unsqueeze_649: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    sub_81: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
    add_204: "f32[56]" = torch.ops.aten.add.Tensor(arg693_1, 1e-05);  arg693_1 = None
    sqrt_81: "f32[56]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
    reciprocal_81: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
    mul_243: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
    unsqueeze_650: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
    unsqueeze_651: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
    mul_244: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
    unsqueeze_652: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
    unsqueeze_653: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
    mul_245: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
    unsqueeze_654: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
    unsqueeze_655: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
    add_205: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_78: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_205);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_206: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_78, getitem_631);  getitem_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_82: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_206, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_206 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_656: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg695_1, -1);  arg695_1 = None
    unsqueeze_657: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    sub_82: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_657);  convolution_82 = unsqueeze_657 = None
    add_207: "f32[56]" = torch.ops.aten.add.Tensor(arg696_1, 1e-05);  arg696_1 = None
    sqrt_82: "f32[56]" = torch.ops.aten.sqrt.default(add_207);  add_207 = None
    reciprocal_82: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
    mul_246: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
    unsqueeze_658: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
    unsqueeze_659: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    mul_247: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
    unsqueeze_660: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
    unsqueeze_661: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
    mul_248: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
    unsqueeze_662: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
    unsqueeze_663: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
    add_208: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_79: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_208);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_209: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_79, getitem_640);  getitem_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_83: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_209, arg249_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_209 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_664: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg698_1, -1);  arg698_1 = None
    unsqueeze_665: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
    sub_83: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_665);  convolution_83 = unsqueeze_665 = None
    add_210: "f32[56]" = torch.ops.aten.add.Tensor(arg699_1, 1e-05);  arg699_1 = None
    sqrt_83: "f32[56]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
    reciprocal_83: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
    mul_249: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
    unsqueeze_666: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
    unsqueeze_667: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    mul_250: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
    unsqueeze_668: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
    unsqueeze_669: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    mul_251: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
    unsqueeze_670: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
    unsqueeze_671: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
    add_211: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_80: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_8: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_74, relu_75, relu_76, relu_77, relu_78, relu_79, relu_80, getitem_649], 1);  relu_74 = relu_75 = relu_76 = relu_77 = relu_78 = relu_79 = relu_80 = getitem_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_84: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_8, arg252_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_8 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_672: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg701_1, -1);  arg701_1 = None
    unsqueeze_673: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
    sub_84: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_673);  convolution_84 = unsqueeze_673 = None
    add_212: "f32[1024]" = torch.ops.aten.add.Tensor(arg702_1, 1e-05);  arg702_1 = None
    sqrt_84: "f32[1024]" = torch.ops.aten.sqrt.default(add_212);  add_212 = None
    reciprocal_84: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
    mul_252: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
    unsqueeze_674: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
    unsqueeze_675: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
    mul_253: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
    unsqueeze_676: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
    unsqueeze_677: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    mul_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
    unsqueeze_678: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
    unsqueeze_679: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    add_213: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_214: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_213, relu_72);  add_213 = relu_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_81: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_214);  add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_85: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_81, arg255_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_680: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
    unsqueeze_681: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
    sub_85: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_681);  convolution_85 = unsqueeze_681 = None
    add_215: "f32[448]" = torch.ops.aten.add.Tensor(arg705_1, 1e-05);  arg705_1 = None
    sqrt_85: "f32[448]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
    reciprocal_85: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
    mul_255: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
    unsqueeze_682: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
    unsqueeze_683: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
    mul_256: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
    unsqueeze_684: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
    unsqueeze_685: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
    mul_257: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
    unsqueeze_686: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
    unsqueeze_687: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
    add_216: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_82: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_82 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_658: "f32[8, 56, 14, 14]" = split_with_sizes_82[0];  split_with_sizes_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_83 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_667: "f32[8, 56, 14, 14]" = split_with_sizes_83[1];  split_with_sizes_83 = None
    split_with_sizes_84 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_676: "f32[8, 56, 14, 14]" = split_with_sizes_84[2];  split_with_sizes_84 = None
    split_with_sizes_85 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_685: "f32[8, 56, 14, 14]" = split_with_sizes_85[3];  split_with_sizes_85 = None
    split_with_sizes_86 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_694: "f32[8, 56, 14, 14]" = split_with_sizes_86[4];  split_with_sizes_86 = None
    split_with_sizes_87 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_703: "f32[8, 56, 14, 14]" = split_with_sizes_87[5];  split_with_sizes_87 = None
    split_with_sizes_88 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_712: "f32[8, 56, 14, 14]" = split_with_sizes_88[6];  split_with_sizes_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_89 = torch.ops.aten.split_with_sizes.default(relu_82, [56, 56, 56, 56, 56, 56, 56, 56], 1);  relu_82 = None
    getitem_721: "f32[8, 56, 14, 14]" = split_with_sizes_89[7];  split_with_sizes_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_86: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_658, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_658 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_688: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg707_1, -1);  arg707_1 = None
    unsqueeze_689: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
    sub_86: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_689);  convolution_86 = unsqueeze_689 = None
    add_217: "f32[56]" = torch.ops.aten.add.Tensor(arg708_1, 1e-05);  arg708_1 = None
    sqrt_86: "f32[56]" = torch.ops.aten.sqrt.default(add_217);  add_217 = None
    reciprocal_86: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
    mul_258: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
    unsqueeze_690: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
    unsqueeze_691: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
    mul_259: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
    unsqueeze_692: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
    unsqueeze_693: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
    mul_260: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
    unsqueeze_694: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
    unsqueeze_695: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
    add_218: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_83: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_218);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_219: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_83, getitem_667);  getitem_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_87: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_219, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_219 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_696: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
    unsqueeze_697: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
    sub_87: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_697);  convolution_87 = unsqueeze_697 = None
    add_220: "f32[56]" = torch.ops.aten.add.Tensor(arg711_1, 1e-05);  arg711_1 = None
    sqrt_87: "f32[56]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
    reciprocal_87: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
    mul_261: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
    unsqueeze_698: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
    unsqueeze_699: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
    mul_262: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
    unsqueeze_700: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
    unsqueeze_701: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
    mul_263: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
    unsqueeze_702: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
    unsqueeze_703: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
    add_221: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_84: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_222: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_84, getitem_676);  getitem_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_88: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_222, arg264_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_222 = arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_704: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg713_1, -1);  arg713_1 = None
    unsqueeze_705: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
    sub_88: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_705);  convolution_88 = unsqueeze_705 = None
    add_223: "f32[56]" = torch.ops.aten.add.Tensor(arg714_1, 1e-05);  arg714_1 = None
    sqrt_88: "f32[56]" = torch.ops.aten.sqrt.default(add_223);  add_223 = None
    reciprocal_88: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
    mul_264: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
    unsqueeze_706: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
    unsqueeze_707: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
    mul_265: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
    unsqueeze_708: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
    unsqueeze_709: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
    mul_266: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
    unsqueeze_710: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
    unsqueeze_711: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
    add_224: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_85: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_224);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_225: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_85, getitem_685);  getitem_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_89: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_225, arg267_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_225 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_712: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg716_1, -1);  arg716_1 = None
    unsqueeze_713: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
    sub_89: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_713);  convolution_89 = unsqueeze_713 = None
    add_226: "f32[56]" = torch.ops.aten.add.Tensor(arg717_1, 1e-05);  arg717_1 = None
    sqrt_89: "f32[56]" = torch.ops.aten.sqrt.default(add_226);  add_226 = None
    reciprocal_89: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
    mul_267: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
    unsqueeze_714: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
    unsqueeze_715: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
    mul_268: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
    unsqueeze_716: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
    unsqueeze_717: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
    mul_269: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
    unsqueeze_718: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
    unsqueeze_719: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
    add_227: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_86: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_227);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_228: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_86, getitem_694);  getitem_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_90: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_228, arg270_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_228 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_720: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
    unsqueeze_721: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
    sub_90: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_721);  convolution_90 = unsqueeze_721 = None
    add_229: "f32[56]" = torch.ops.aten.add.Tensor(arg720_1, 1e-05);  arg720_1 = None
    sqrt_90: "f32[56]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
    reciprocal_90: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
    mul_270: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
    unsqueeze_722: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
    unsqueeze_723: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
    mul_271: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
    unsqueeze_724: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
    unsqueeze_725: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
    mul_272: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
    unsqueeze_726: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
    unsqueeze_727: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
    add_230: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_87: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_230);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_231: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_87, getitem_703);  getitem_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_91: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_231, arg273_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_231 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_728: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg722_1, -1);  arg722_1 = None
    unsqueeze_729: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
    sub_91: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_729);  convolution_91 = unsqueeze_729 = None
    add_232: "f32[56]" = torch.ops.aten.add.Tensor(arg723_1, 1e-05);  arg723_1 = None
    sqrt_91: "f32[56]" = torch.ops.aten.sqrt.default(add_232);  add_232 = None
    reciprocal_91: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
    mul_273: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
    unsqueeze_730: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
    unsqueeze_731: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
    mul_274: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
    unsqueeze_732: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
    unsqueeze_733: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
    mul_275: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
    unsqueeze_734: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
    unsqueeze_735: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
    add_233: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_88: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_233);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_234: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_88, getitem_712);  getitem_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_92: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_234, arg276_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_234 = arg276_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_736: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
    unsqueeze_737: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
    sub_92: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_737);  convolution_92 = unsqueeze_737 = None
    add_235: "f32[56]" = torch.ops.aten.add.Tensor(arg726_1, 1e-05);  arg726_1 = None
    sqrt_92: "f32[56]" = torch.ops.aten.sqrt.default(add_235);  add_235 = None
    reciprocal_92: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
    mul_276: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
    unsqueeze_738: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
    unsqueeze_739: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
    mul_277: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
    unsqueeze_740: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
    unsqueeze_741: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
    mul_278: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
    unsqueeze_742: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
    unsqueeze_743: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
    add_236: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_89: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_236);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_9: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_83, relu_84, relu_85, relu_86, relu_87, relu_88, relu_89, getitem_721], 1);  relu_83 = relu_84 = relu_85 = relu_86 = relu_87 = relu_88 = relu_89 = getitem_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_9, arg279_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_9 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_744: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg728_1, -1);  arg728_1 = None
    unsqueeze_745: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
    sub_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_745);  convolution_93 = unsqueeze_745 = None
    add_237: "f32[1024]" = torch.ops.aten.add.Tensor(arg729_1, 1e-05);  arg729_1 = None
    sqrt_93: "f32[1024]" = torch.ops.aten.sqrt.default(add_237);  add_237 = None
    reciprocal_93: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
    mul_279: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
    unsqueeze_746: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
    unsqueeze_747: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
    mul_280: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
    unsqueeze_748: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
    unsqueeze_749: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
    mul_281: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
    unsqueeze_750: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
    unsqueeze_751: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
    add_238: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_239: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_238, relu_81);  add_238 = relu_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_90: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_239);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_94: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_90, arg282_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_752: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg731_1, -1);  arg731_1 = None
    unsqueeze_753: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
    sub_94: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_753);  convolution_94 = unsqueeze_753 = None
    add_240: "f32[448]" = torch.ops.aten.add.Tensor(arg732_1, 1e-05);  arg732_1 = None
    sqrt_94: "f32[448]" = torch.ops.aten.sqrt.default(add_240);  add_240 = None
    reciprocal_94: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
    mul_282: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
    unsqueeze_754: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
    unsqueeze_755: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
    mul_283: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
    unsqueeze_756: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
    unsqueeze_757: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
    mul_284: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
    unsqueeze_758: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
    unsqueeze_759: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
    add_241: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_91: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_241);  add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_91 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_730: "f32[8, 56, 14, 14]" = split_with_sizes_91[0];  split_with_sizes_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_92 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_739: "f32[8, 56, 14, 14]" = split_with_sizes_92[1];  split_with_sizes_92 = None
    split_with_sizes_93 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_748: "f32[8, 56, 14, 14]" = split_with_sizes_93[2];  split_with_sizes_93 = None
    split_with_sizes_94 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_757: "f32[8, 56, 14, 14]" = split_with_sizes_94[3];  split_with_sizes_94 = None
    split_with_sizes_95 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_766: "f32[8, 56, 14, 14]" = split_with_sizes_95[4];  split_with_sizes_95 = None
    split_with_sizes_96 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_775: "f32[8, 56, 14, 14]" = split_with_sizes_96[5];  split_with_sizes_96 = None
    split_with_sizes_97 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_784: "f32[8, 56, 14, 14]" = split_with_sizes_97[6];  split_with_sizes_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_98 = torch.ops.aten.split_with_sizes.default(relu_91, [56, 56, 56, 56, 56, 56, 56, 56], 1);  relu_91 = None
    getitem_793: "f32[8, 56, 14, 14]" = split_with_sizes_98[7];  split_with_sizes_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_95: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_730, arg285_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_730 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_760: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg734_1, -1);  arg734_1 = None
    unsqueeze_761: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
    sub_95: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_761);  convolution_95 = unsqueeze_761 = None
    add_242: "f32[56]" = torch.ops.aten.add.Tensor(arg735_1, 1e-05);  arg735_1 = None
    sqrt_95: "f32[56]" = torch.ops.aten.sqrt.default(add_242);  add_242 = None
    reciprocal_95: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
    mul_285: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
    unsqueeze_762: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
    unsqueeze_763: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
    mul_286: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
    unsqueeze_764: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg286_1, -1);  arg286_1 = None
    unsqueeze_765: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
    mul_287: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
    unsqueeze_766: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
    unsqueeze_767: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
    add_243: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_92: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_243);  add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_244: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_92, getitem_739);  getitem_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_96: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_244, arg288_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_244 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_768: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg737_1, -1);  arg737_1 = None
    unsqueeze_769: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
    sub_96: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_769);  convolution_96 = unsqueeze_769 = None
    add_245: "f32[56]" = torch.ops.aten.add.Tensor(arg738_1, 1e-05);  arg738_1 = None
    sqrt_96: "f32[56]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
    reciprocal_96: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
    mul_288: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
    unsqueeze_770: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
    unsqueeze_771: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
    mul_289: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
    unsqueeze_772: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
    unsqueeze_773: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
    mul_290: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
    unsqueeze_774: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
    unsqueeze_775: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
    add_246: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_93: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_246);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_247: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_93, getitem_748);  getitem_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_97: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_247, arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_247 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_776: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
    unsqueeze_777: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
    sub_97: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_777);  convolution_97 = unsqueeze_777 = None
    add_248: "f32[56]" = torch.ops.aten.add.Tensor(arg741_1, 1e-05);  arg741_1 = None
    sqrt_97: "f32[56]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
    reciprocal_97: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
    mul_291: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
    unsqueeze_778: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
    unsqueeze_779: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
    mul_292: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
    unsqueeze_780: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
    unsqueeze_781: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
    mul_293: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
    unsqueeze_782: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
    unsqueeze_783: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
    add_249: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_94: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_249);  add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_250: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_94, getitem_757);  getitem_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_98: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_250, arg294_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_250 = arg294_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_784: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg743_1, -1);  arg743_1 = None
    unsqueeze_785: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
    sub_98: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_785);  convolution_98 = unsqueeze_785 = None
    add_251: "f32[56]" = torch.ops.aten.add.Tensor(arg744_1, 1e-05);  arg744_1 = None
    sqrt_98: "f32[56]" = torch.ops.aten.sqrt.default(add_251);  add_251 = None
    reciprocal_98: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
    mul_294: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
    unsqueeze_786: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
    unsqueeze_787: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
    mul_295: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
    unsqueeze_788: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
    unsqueeze_789: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
    mul_296: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
    unsqueeze_790: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
    unsqueeze_791: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
    add_252: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_95: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_252);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_253: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_95, getitem_766);  getitem_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_99: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_253, arg297_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_253 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_792: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg746_1, -1);  arg746_1 = None
    unsqueeze_793: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
    sub_99: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_793);  convolution_99 = unsqueeze_793 = None
    add_254: "f32[56]" = torch.ops.aten.add.Tensor(arg747_1, 1e-05);  arg747_1 = None
    sqrt_99: "f32[56]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
    reciprocal_99: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
    mul_297: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
    unsqueeze_794: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
    unsqueeze_795: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
    mul_298: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
    unsqueeze_796: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
    unsqueeze_797: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
    mul_299: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
    unsqueeze_798: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
    unsqueeze_799: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
    add_255: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_96: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_255);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_256: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_96, getitem_775);  getitem_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_100: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_256, arg300_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_256 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_800: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg749_1, -1);  arg749_1 = None
    unsqueeze_801: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
    sub_100: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_801);  convolution_100 = unsqueeze_801 = None
    add_257: "f32[56]" = torch.ops.aten.add.Tensor(arg750_1, 1e-05);  arg750_1 = None
    sqrt_100: "f32[56]" = torch.ops.aten.sqrt.default(add_257);  add_257 = None
    reciprocal_100: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
    mul_300: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
    unsqueeze_802: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
    unsqueeze_803: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
    mul_301: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
    unsqueeze_804: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
    unsqueeze_805: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
    mul_302: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
    unsqueeze_806: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
    unsqueeze_807: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
    add_258: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_97: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_258);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_259: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_97, getitem_784);  getitem_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_101: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_259, arg303_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_259 = arg303_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_808: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg752_1, -1);  arg752_1 = None
    unsqueeze_809: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
    sub_101: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_809);  convolution_101 = unsqueeze_809 = None
    add_260: "f32[56]" = torch.ops.aten.add.Tensor(arg753_1, 1e-05);  arg753_1 = None
    sqrt_101: "f32[56]" = torch.ops.aten.sqrt.default(add_260);  add_260 = None
    reciprocal_101: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
    mul_303: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
    unsqueeze_810: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
    unsqueeze_811: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
    mul_304: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
    unsqueeze_812: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
    unsqueeze_813: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
    mul_305: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
    unsqueeze_814: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
    unsqueeze_815: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
    add_261: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_98: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_261);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_10: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_92, relu_93, relu_94, relu_95, relu_96, relu_97, relu_98, getitem_793], 1);  relu_92 = relu_93 = relu_94 = relu_95 = relu_96 = relu_97 = relu_98 = getitem_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_102: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_10, arg306_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_10 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_816: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg755_1, -1);  arg755_1 = None
    unsqueeze_817: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
    sub_102: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_817);  convolution_102 = unsqueeze_817 = None
    add_262: "f32[1024]" = torch.ops.aten.add.Tensor(arg756_1, 1e-05);  arg756_1 = None
    sqrt_102: "f32[1024]" = torch.ops.aten.sqrt.default(add_262);  add_262 = None
    reciprocal_102: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
    mul_306: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
    unsqueeze_818: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
    unsqueeze_819: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
    mul_307: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
    unsqueeze_820: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
    unsqueeze_821: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
    mul_308: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
    unsqueeze_822: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
    unsqueeze_823: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
    add_263: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_263, relu_90);  add_263 = relu_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_99: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_264);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_103: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_99, arg309_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_824: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg758_1, -1);  arg758_1 = None
    unsqueeze_825: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
    sub_103: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_825);  convolution_103 = unsqueeze_825 = None
    add_265: "f32[448]" = torch.ops.aten.add.Tensor(arg759_1, 1e-05);  arg759_1 = None
    sqrt_103: "f32[448]" = torch.ops.aten.sqrt.default(add_265);  add_265 = None
    reciprocal_103: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
    mul_309: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
    unsqueeze_826: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
    unsqueeze_827: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
    mul_310: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
    unsqueeze_828: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
    unsqueeze_829: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
    mul_311: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
    unsqueeze_830: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
    unsqueeze_831: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
    add_266: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_100: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_266);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_100 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_802: "f32[8, 56, 14, 14]" = split_with_sizes_100[0];  split_with_sizes_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_101 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_811: "f32[8, 56, 14, 14]" = split_with_sizes_101[1];  split_with_sizes_101 = None
    split_with_sizes_102 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_820: "f32[8, 56, 14, 14]" = split_with_sizes_102[2];  split_with_sizes_102 = None
    split_with_sizes_103 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_829: "f32[8, 56, 14, 14]" = split_with_sizes_103[3];  split_with_sizes_103 = None
    split_with_sizes_104 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_838: "f32[8, 56, 14, 14]" = split_with_sizes_104[4];  split_with_sizes_104 = None
    split_with_sizes_105 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_847: "f32[8, 56, 14, 14]" = split_with_sizes_105[5];  split_with_sizes_105 = None
    split_with_sizes_106 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_856: "f32[8, 56, 14, 14]" = split_with_sizes_106[6];  split_with_sizes_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_107 = torch.ops.aten.split_with_sizes.default(relu_100, [56, 56, 56, 56, 56, 56, 56, 56], 1);  relu_100 = None
    getitem_865: "f32[8, 56, 14, 14]" = split_with_sizes_107[7];  split_with_sizes_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_104: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_802, arg312_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_802 = arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_832: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg761_1, -1);  arg761_1 = None
    unsqueeze_833: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
    sub_104: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_833);  convolution_104 = unsqueeze_833 = None
    add_267: "f32[56]" = torch.ops.aten.add.Tensor(arg762_1, 1e-05);  arg762_1 = None
    sqrt_104: "f32[56]" = torch.ops.aten.sqrt.default(add_267);  add_267 = None
    reciprocal_104: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
    mul_312: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
    unsqueeze_834: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
    unsqueeze_835: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
    mul_313: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
    unsqueeze_836: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
    unsqueeze_837: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
    mul_314: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
    unsqueeze_838: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
    unsqueeze_839: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
    add_268: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_101: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_268);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_269: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_101, getitem_811);  getitem_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_105: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_269, arg315_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_269 = arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_840: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg764_1, -1);  arg764_1 = None
    unsqueeze_841: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
    sub_105: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_841);  convolution_105 = unsqueeze_841 = None
    add_270: "f32[56]" = torch.ops.aten.add.Tensor(arg765_1, 1e-05);  arg765_1 = None
    sqrt_105: "f32[56]" = torch.ops.aten.sqrt.default(add_270);  add_270 = None
    reciprocal_105: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
    mul_315: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
    unsqueeze_842: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
    unsqueeze_843: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
    mul_316: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
    unsqueeze_844: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
    unsqueeze_845: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
    mul_317: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
    unsqueeze_846: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
    unsqueeze_847: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
    add_271: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_102: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_271);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_272: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_102, getitem_820);  getitem_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_106: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_272, arg318_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_272 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_848: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg767_1, -1);  arg767_1 = None
    unsqueeze_849: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
    sub_106: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_849);  convolution_106 = unsqueeze_849 = None
    add_273: "f32[56]" = torch.ops.aten.add.Tensor(arg768_1, 1e-05);  arg768_1 = None
    sqrt_106: "f32[56]" = torch.ops.aten.sqrt.default(add_273);  add_273 = None
    reciprocal_106: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
    mul_318: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
    unsqueeze_850: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
    unsqueeze_851: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
    mul_319: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
    unsqueeze_852: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
    unsqueeze_853: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
    mul_320: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
    unsqueeze_854: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
    unsqueeze_855: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
    add_274: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_103: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_274);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_275: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_103, getitem_829);  getitem_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_107: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_275, arg321_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_275 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_856: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg770_1, -1);  arg770_1 = None
    unsqueeze_857: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
    sub_107: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_857);  convolution_107 = unsqueeze_857 = None
    add_276: "f32[56]" = torch.ops.aten.add.Tensor(arg771_1, 1e-05);  arg771_1 = None
    sqrt_107: "f32[56]" = torch.ops.aten.sqrt.default(add_276);  add_276 = None
    reciprocal_107: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
    mul_321: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
    unsqueeze_858: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
    unsqueeze_859: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
    mul_322: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
    unsqueeze_860: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
    unsqueeze_861: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
    mul_323: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
    unsqueeze_862: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
    unsqueeze_863: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
    add_277: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_104: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_277);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_278: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_104, getitem_838);  getitem_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_108: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_278, arg324_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_278 = arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_864: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg773_1, -1);  arg773_1 = None
    unsqueeze_865: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
    sub_108: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_865);  convolution_108 = unsqueeze_865 = None
    add_279: "f32[56]" = torch.ops.aten.add.Tensor(arg774_1, 1e-05);  arg774_1 = None
    sqrt_108: "f32[56]" = torch.ops.aten.sqrt.default(add_279);  add_279 = None
    reciprocal_108: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
    mul_324: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
    unsqueeze_866: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
    unsqueeze_867: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
    mul_325: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
    unsqueeze_868: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
    unsqueeze_869: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
    mul_326: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
    unsqueeze_870: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
    unsqueeze_871: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
    add_280: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_105: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_280);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_281: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_105, getitem_847);  getitem_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_109: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_281, arg327_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_281 = arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_872: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg776_1, -1);  arg776_1 = None
    unsqueeze_873: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
    sub_109: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_873);  convolution_109 = unsqueeze_873 = None
    add_282: "f32[56]" = torch.ops.aten.add.Tensor(arg777_1, 1e-05);  arg777_1 = None
    sqrt_109: "f32[56]" = torch.ops.aten.sqrt.default(add_282);  add_282 = None
    reciprocal_109: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
    mul_327: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
    unsqueeze_874: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
    unsqueeze_875: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
    mul_328: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
    unsqueeze_876: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
    unsqueeze_877: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
    mul_329: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
    unsqueeze_878: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
    unsqueeze_879: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
    add_283: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_106: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_283);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_284: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_106, getitem_856);  getitem_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_110: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_284, arg330_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_284 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_880: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg779_1, -1);  arg779_1 = None
    unsqueeze_881: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
    sub_110: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_881);  convolution_110 = unsqueeze_881 = None
    add_285: "f32[56]" = torch.ops.aten.add.Tensor(arg780_1, 1e-05);  arg780_1 = None
    sqrt_110: "f32[56]" = torch.ops.aten.sqrt.default(add_285);  add_285 = None
    reciprocal_110: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
    mul_330: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
    unsqueeze_882: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
    unsqueeze_883: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
    mul_331: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
    unsqueeze_884: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
    unsqueeze_885: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
    mul_332: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
    unsqueeze_886: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
    unsqueeze_887: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
    add_286: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_107: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_286);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_11: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_101, relu_102, relu_103, relu_104, relu_105, relu_106, relu_107, getitem_865], 1);  relu_101 = relu_102 = relu_103 = relu_104 = relu_105 = relu_106 = relu_107 = getitem_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_111: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_11, arg333_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_11 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_888: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg782_1, -1);  arg782_1 = None
    unsqueeze_889: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
    sub_111: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
    add_287: "f32[1024]" = torch.ops.aten.add.Tensor(arg783_1, 1e-05);  arg783_1 = None
    sqrt_111: "f32[1024]" = torch.ops.aten.sqrt.default(add_287);  add_287 = None
    reciprocal_111: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
    mul_333: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
    unsqueeze_890: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
    unsqueeze_891: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
    mul_334: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
    unsqueeze_892: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
    unsqueeze_893: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
    mul_335: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
    unsqueeze_894: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
    unsqueeze_895: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
    add_288: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_289: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_288, relu_99);  add_288 = relu_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_108: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_289);  add_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_112: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_108, arg336_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg336_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_896: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg785_1, -1);  arg785_1 = None
    unsqueeze_897: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
    sub_112: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_897);  convolution_112 = unsqueeze_897 = None
    add_290: "f32[448]" = torch.ops.aten.add.Tensor(arg786_1, 1e-05);  arg786_1 = None
    sqrt_112: "f32[448]" = torch.ops.aten.sqrt.default(add_290);  add_290 = None
    reciprocal_112: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
    mul_336: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
    unsqueeze_898: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
    unsqueeze_899: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
    mul_337: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
    unsqueeze_900: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
    unsqueeze_901: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
    mul_338: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
    unsqueeze_902: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
    unsqueeze_903: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
    add_291: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_109: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_291);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_109 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_874: "f32[8, 56, 14, 14]" = split_with_sizes_109[0];  split_with_sizes_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_110 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_883: "f32[8, 56, 14, 14]" = split_with_sizes_110[1];  split_with_sizes_110 = None
    split_with_sizes_111 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_892: "f32[8, 56, 14, 14]" = split_with_sizes_111[2];  split_with_sizes_111 = None
    split_with_sizes_112 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_901: "f32[8, 56, 14, 14]" = split_with_sizes_112[3];  split_with_sizes_112 = None
    split_with_sizes_113 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_910: "f32[8, 56, 14, 14]" = split_with_sizes_113[4];  split_with_sizes_113 = None
    split_with_sizes_114 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_919: "f32[8, 56, 14, 14]" = split_with_sizes_114[5];  split_with_sizes_114 = None
    split_with_sizes_115 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1)
    getitem_928: "f32[8, 56, 14, 14]" = split_with_sizes_115[6];  split_with_sizes_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_116 = torch.ops.aten.split_with_sizes.default(relu_109, [56, 56, 56, 56, 56, 56, 56, 56], 1);  relu_109 = None
    getitem_937: "f32[8, 56, 14, 14]" = split_with_sizes_116[7];  split_with_sizes_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_113: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_874, arg339_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_874 = arg339_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_904: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg788_1, -1);  arg788_1 = None
    unsqueeze_905: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
    sub_113: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_905);  convolution_113 = unsqueeze_905 = None
    add_292: "f32[56]" = torch.ops.aten.add.Tensor(arg789_1, 1e-05);  arg789_1 = None
    sqrt_113: "f32[56]" = torch.ops.aten.sqrt.default(add_292);  add_292 = None
    reciprocal_113: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
    mul_339: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
    unsqueeze_906: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
    unsqueeze_907: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
    mul_340: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
    unsqueeze_908: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
    unsqueeze_909: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
    mul_341: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
    unsqueeze_910: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
    unsqueeze_911: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
    add_293: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_110: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_293);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_294: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_110, getitem_883);  getitem_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_114: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_294, arg342_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_294 = arg342_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_912: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg791_1, -1);  arg791_1 = None
    unsqueeze_913: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
    sub_114: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_913);  convolution_114 = unsqueeze_913 = None
    add_295: "f32[56]" = torch.ops.aten.add.Tensor(arg792_1, 1e-05);  arg792_1 = None
    sqrt_114: "f32[56]" = torch.ops.aten.sqrt.default(add_295);  add_295 = None
    reciprocal_114: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
    mul_342: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
    unsqueeze_914: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
    unsqueeze_915: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
    mul_343: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
    unsqueeze_916: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
    unsqueeze_917: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
    mul_344: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
    unsqueeze_918: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
    unsqueeze_919: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
    add_296: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_111: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_296);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_297: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_111, getitem_892);  getitem_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_115: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_297, arg345_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_297 = arg345_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_920: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg794_1, -1);  arg794_1 = None
    unsqueeze_921: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
    sub_115: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_921);  convolution_115 = unsqueeze_921 = None
    add_298: "f32[56]" = torch.ops.aten.add.Tensor(arg795_1, 1e-05);  arg795_1 = None
    sqrt_115: "f32[56]" = torch.ops.aten.sqrt.default(add_298);  add_298 = None
    reciprocal_115: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
    mul_345: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
    unsqueeze_922: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
    unsqueeze_923: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
    mul_346: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
    unsqueeze_924: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
    unsqueeze_925: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
    mul_347: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
    unsqueeze_926: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
    unsqueeze_927: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
    add_299: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_112: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_299);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_300: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_112, getitem_901);  getitem_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_116: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_300, arg348_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_300 = arg348_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_928: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg797_1, -1);  arg797_1 = None
    unsqueeze_929: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
    sub_116: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_929);  convolution_116 = unsqueeze_929 = None
    add_301: "f32[56]" = torch.ops.aten.add.Tensor(arg798_1, 1e-05);  arg798_1 = None
    sqrt_116: "f32[56]" = torch.ops.aten.sqrt.default(add_301);  add_301 = None
    reciprocal_116: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
    mul_348: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
    unsqueeze_930: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
    unsqueeze_931: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
    mul_349: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
    unsqueeze_932: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
    unsqueeze_933: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
    mul_350: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
    unsqueeze_934: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
    unsqueeze_935: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
    add_302: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_113: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_302);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_303: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_113, getitem_910);  getitem_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_117: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_303, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_303 = arg351_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_936: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg800_1, -1);  arg800_1 = None
    unsqueeze_937: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
    sub_117: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_937);  convolution_117 = unsqueeze_937 = None
    add_304: "f32[56]" = torch.ops.aten.add.Tensor(arg801_1, 1e-05);  arg801_1 = None
    sqrt_117: "f32[56]" = torch.ops.aten.sqrt.default(add_304);  add_304 = None
    reciprocal_117: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
    mul_351: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
    unsqueeze_938: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
    unsqueeze_939: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
    mul_352: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
    unsqueeze_940: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
    unsqueeze_941: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
    mul_353: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
    unsqueeze_942: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg353_1, -1);  arg353_1 = None
    unsqueeze_943: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
    add_305: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_114: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_305);  add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_306: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_114, getitem_919);  getitem_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_118: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_306, arg354_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_306 = arg354_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_944: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg803_1, -1);  arg803_1 = None
    unsqueeze_945: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
    sub_118: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_945);  convolution_118 = unsqueeze_945 = None
    add_307: "f32[56]" = torch.ops.aten.add.Tensor(arg804_1, 1e-05);  arg804_1 = None
    sqrt_118: "f32[56]" = torch.ops.aten.sqrt.default(add_307);  add_307 = None
    reciprocal_118: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
    mul_354: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
    unsqueeze_946: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
    unsqueeze_947: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
    mul_355: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
    unsqueeze_948: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
    unsqueeze_949: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
    mul_356: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
    unsqueeze_950: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg356_1, -1);  arg356_1 = None
    unsqueeze_951: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
    add_308: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_115: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_308);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_309: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_115, getitem_928);  getitem_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_119: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_309, arg357_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_309 = arg357_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_952: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg806_1, -1);  arg806_1 = None
    unsqueeze_953: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
    sub_119: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_953);  convolution_119 = unsqueeze_953 = None
    add_310: "f32[56]" = torch.ops.aten.add.Tensor(arg807_1, 1e-05);  arg807_1 = None
    sqrt_119: "f32[56]" = torch.ops.aten.sqrt.default(add_310);  add_310 = None
    reciprocal_119: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
    mul_357: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
    unsqueeze_954: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
    unsqueeze_955: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
    mul_358: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
    unsqueeze_956: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
    unsqueeze_957: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
    mul_359: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
    unsqueeze_958: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
    unsqueeze_959: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
    add_311: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_116: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_311);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_12: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_110, relu_111, relu_112, relu_113, relu_114, relu_115, relu_116, getitem_937], 1);  relu_110 = relu_111 = relu_112 = relu_113 = relu_114 = relu_115 = relu_116 = getitem_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_120: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_12, arg360_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_12 = arg360_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_960: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg809_1, -1);  arg809_1 = None
    unsqueeze_961: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
    sub_120: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_961);  convolution_120 = unsqueeze_961 = None
    add_312: "f32[1024]" = torch.ops.aten.add.Tensor(arg810_1, 1e-05);  arg810_1 = None
    sqrt_120: "f32[1024]" = torch.ops.aten.sqrt.default(add_312);  add_312 = None
    reciprocal_120: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
    mul_360: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
    unsqueeze_962: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
    unsqueeze_963: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
    mul_361: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
    unsqueeze_964: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
    unsqueeze_965: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
    mul_362: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
    unsqueeze_966: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
    unsqueeze_967: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
    add_313: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_314: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_313, relu_108);  add_313 = relu_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_117: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_314);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_121: "f32[8, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_117, arg363_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_968: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg812_1, -1);  arg812_1 = None
    unsqueeze_969: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
    sub_121: "f32[8, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_969);  convolution_121 = unsqueeze_969 = None
    add_315: "f32[896]" = torch.ops.aten.add.Tensor(arg813_1, 1e-05);  arg813_1 = None
    sqrt_121: "f32[896]" = torch.ops.aten.sqrt.default(add_315);  add_315 = None
    reciprocal_121: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
    mul_363: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
    unsqueeze_970: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
    unsqueeze_971: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
    mul_364: "f32[8, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
    unsqueeze_972: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
    unsqueeze_973: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
    mul_365: "f32[8, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
    unsqueeze_974: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
    unsqueeze_975: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
    add_316: "f32[8, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_118: "f32[8, 896, 14, 14]" = torch.ops.aten.relu.default(add_316);  add_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_118 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_946: "f32[8, 112, 14, 14]" = split_with_sizes_118[0];  split_with_sizes_118 = None
    split_with_sizes_119 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_955: "f32[8, 112, 14, 14]" = split_with_sizes_119[1];  split_with_sizes_119 = None
    split_with_sizes_120 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_964: "f32[8, 112, 14, 14]" = split_with_sizes_120[2];  split_with_sizes_120 = None
    split_with_sizes_121 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_973: "f32[8, 112, 14, 14]" = split_with_sizes_121[3];  split_with_sizes_121 = None
    split_with_sizes_122 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_982: "f32[8, 112, 14, 14]" = split_with_sizes_122[4];  split_with_sizes_122 = None
    split_with_sizes_123 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_991: "f32[8, 112, 14, 14]" = split_with_sizes_123[5];  split_with_sizes_123 = None
    split_with_sizes_124 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1000: "f32[8, 112, 14, 14]" = split_with_sizes_124[6];  split_with_sizes_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_125 = torch.ops.aten.split_with_sizes.default(relu_118, [112, 112, 112, 112, 112, 112, 112, 112], 1);  relu_118 = None
    getitem_1009: "f32[8, 112, 14, 14]" = split_with_sizes_125[7];  split_with_sizes_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_122: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_946, arg366_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_946 = arg366_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_976: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg815_1, -1);  arg815_1 = None
    unsqueeze_977: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
    sub_122: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_977);  convolution_122 = unsqueeze_977 = None
    add_317: "f32[112]" = torch.ops.aten.add.Tensor(arg816_1, 1e-05);  arg816_1 = None
    sqrt_122: "f32[112]" = torch.ops.aten.sqrt.default(add_317);  add_317 = None
    reciprocal_122: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
    mul_366: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
    unsqueeze_978: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
    unsqueeze_979: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
    mul_367: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
    unsqueeze_980: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
    unsqueeze_981: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
    mul_368: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_981);  mul_367 = unsqueeze_981 = None
    unsqueeze_982: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
    unsqueeze_983: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
    add_318: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_368, unsqueeze_983);  mul_368 = unsqueeze_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_119: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_318);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_123: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_955, arg369_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_955 = arg369_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_984: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg818_1, -1);  arg818_1 = None
    unsqueeze_985: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
    sub_123: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_985);  convolution_123 = unsqueeze_985 = None
    add_319: "f32[112]" = torch.ops.aten.add.Tensor(arg819_1, 1e-05);  arg819_1 = None
    sqrt_123: "f32[112]" = torch.ops.aten.sqrt.default(add_319);  add_319 = None
    reciprocal_123: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
    mul_369: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
    unsqueeze_986: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
    unsqueeze_987: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
    mul_370: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
    unsqueeze_988: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
    unsqueeze_989: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
    mul_371: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_989);  mul_370 = unsqueeze_989 = None
    unsqueeze_990: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
    unsqueeze_991: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
    add_320: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_991);  mul_371 = unsqueeze_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_120: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_320);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_124: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_964, arg372_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_964 = arg372_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_992: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg821_1, -1);  arg821_1 = None
    unsqueeze_993: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
    sub_124: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_993);  convolution_124 = unsqueeze_993 = None
    add_321: "f32[112]" = torch.ops.aten.add.Tensor(arg822_1, 1e-05);  arg822_1 = None
    sqrt_124: "f32[112]" = torch.ops.aten.sqrt.default(add_321);  add_321 = None
    reciprocal_124: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
    mul_372: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
    unsqueeze_994: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
    unsqueeze_995: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
    mul_373: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
    unsqueeze_996: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
    unsqueeze_997: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
    mul_374: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_997);  mul_373 = unsqueeze_997 = None
    unsqueeze_998: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
    unsqueeze_999: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
    add_322: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_999);  mul_374 = unsqueeze_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_121: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_322);  add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_125: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_973, arg375_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_973 = arg375_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1000: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg824_1, -1);  arg824_1 = None
    unsqueeze_1001: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
    sub_125: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1001);  convolution_125 = unsqueeze_1001 = None
    add_323: "f32[112]" = torch.ops.aten.add.Tensor(arg825_1, 1e-05);  arg825_1 = None
    sqrt_125: "f32[112]" = torch.ops.aten.sqrt.default(add_323);  add_323 = None
    reciprocal_125: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
    mul_375: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
    unsqueeze_1002: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
    unsqueeze_1003: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
    mul_376: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
    unsqueeze_1004: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
    unsqueeze_1005: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
    mul_377: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_1005);  mul_376 = unsqueeze_1005 = None
    unsqueeze_1006: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
    unsqueeze_1007: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
    add_324: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_1007);  mul_377 = unsqueeze_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_122: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_324);  add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_126: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_982, arg378_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_982 = arg378_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1008: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg827_1, -1);  arg827_1 = None
    unsqueeze_1009: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
    sub_126: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1009);  convolution_126 = unsqueeze_1009 = None
    add_325: "f32[112]" = torch.ops.aten.add.Tensor(arg828_1, 1e-05);  arg828_1 = None
    sqrt_126: "f32[112]" = torch.ops.aten.sqrt.default(add_325);  add_325 = None
    reciprocal_126: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
    mul_378: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
    unsqueeze_1010: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
    unsqueeze_1011: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
    mul_379: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
    unsqueeze_1012: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
    unsqueeze_1013: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
    mul_380: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_1013);  mul_379 = unsqueeze_1013 = None
    unsqueeze_1014: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
    unsqueeze_1015: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
    add_326: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_1015);  mul_380 = unsqueeze_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_123: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_326);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_127: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_991, arg381_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_991 = arg381_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1016: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg830_1, -1);  arg830_1 = None
    unsqueeze_1017: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
    sub_127: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1017);  convolution_127 = unsqueeze_1017 = None
    add_327: "f32[112]" = torch.ops.aten.add.Tensor(arg831_1, 1e-05);  arg831_1 = None
    sqrt_127: "f32[112]" = torch.ops.aten.sqrt.default(add_327);  add_327 = None
    reciprocal_127: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
    mul_381: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
    unsqueeze_1018: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
    unsqueeze_1019: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
    mul_382: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
    unsqueeze_1020: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
    unsqueeze_1021: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
    mul_383: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_1021);  mul_382 = unsqueeze_1021 = None
    unsqueeze_1022: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg383_1, -1);  arg383_1 = None
    unsqueeze_1023: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
    add_328: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_1023);  mul_383 = unsqueeze_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_124: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_328);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_128: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_1000, arg384_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1000 = arg384_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1024: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg833_1, -1);  arg833_1 = None
    unsqueeze_1025: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
    sub_128: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_1025);  convolution_128 = unsqueeze_1025 = None
    add_329: "f32[112]" = torch.ops.aten.add.Tensor(arg834_1, 1e-05);  arg834_1 = None
    sqrt_128: "f32[112]" = torch.ops.aten.sqrt.default(add_329);  add_329 = None
    reciprocal_128: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
    mul_384: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
    unsqueeze_1026: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
    unsqueeze_1027: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
    mul_385: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
    unsqueeze_1028: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
    unsqueeze_1029: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
    mul_386: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1029);  mul_385 = unsqueeze_1029 = None
    unsqueeze_1030: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
    unsqueeze_1031: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
    add_330: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1031);  mul_386 = unsqueeze_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_125: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_330);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_3: "f32[8, 112, 7, 7]" = torch.ops.aten.avg_pool2d.default(getitem_1009, [3, 3], [2, 2], [1, 1]);  getitem_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_13: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([relu_119, relu_120, relu_121, relu_122, relu_123, relu_124, relu_125, avg_pool2d_3], 1);  relu_119 = relu_120 = relu_121 = relu_122 = relu_123 = relu_124 = relu_125 = avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_129: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_13, arg387_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_13 = arg387_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1032: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg836_1, -1);  arg836_1 = None
    unsqueeze_1033: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
    sub_129: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1033);  convolution_129 = unsqueeze_1033 = None
    add_331: "f32[2048]" = torch.ops.aten.add.Tensor(arg837_1, 1e-05);  arg837_1 = None
    sqrt_129: "f32[2048]" = torch.ops.aten.sqrt.default(add_331);  add_331 = None
    reciprocal_129: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
    mul_387: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
    unsqueeze_1034: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
    unsqueeze_1035: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
    mul_388: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
    unsqueeze_1036: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
    unsqueeze_1037: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
    mul_389: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1037);  mul_388 = unsqueeze_1037 = None
    unsqueeze_1038: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
    unsqueeze_1039: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
    add_332: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1039);  mul_389 = unsqueeze_1039 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_130: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_117, arg390_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_117 = arg390_1 = None
    unsqueeze_1040: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg839_1, -1);  arg839_1 = None
    unsqueeze_1041: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
    sub_130: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1041);  convolution_130 = unsqueeze_1041 = None
    add_333: "f32[2048]" = torch.ops.aten.add.Tensor(arg840_1, 1e-05);  arg840_1 = None
    sqrt_130: "f32[2048]" = torch.ops.aten.sqrt.default(add_333);  add_333 = None
    reciprocal_130: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
    mul_390: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
    unsqueeze_1042: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
    unsqueeze_1043: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
    mul_391: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
    unsqueeze_1044: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
    unsqueeze_1045: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
    mul_392: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1045);  mul_391 = unsqueeze_1045 = None
    unsqueeze_1046: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
    unsqueeze_1047: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
    add_334: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1047);  mul_392 = unsqueeze_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_335: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_332, add_334);  add_332 = add_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_126: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_335);  add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_131: "f32[8, 896, 7, 7]" = torch.ops.aten.convolution.default(relu_126, arg393_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg393_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1048: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg842_1, -1);  arg842_1 = None
    unsqueeze_1049: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
    sub_131: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_1049);  convolution_131 = unsqueeze_1049 = None
    add_336: "f32[896]" = torch.ops.aten.add.Tensor(arg843_1, 1e-05);  arg843_1 = None
    sqrt_131: "f32[896]" = torch.ops.aten.sqrt.default(add_336);  add_336 = None
    reciprocal_131: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
    mul_393: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
    unsqueeze_1050: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
    unsqueeze_1051: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
    mul_394: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
    unsqueeze_1052: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
    unsqueeze_1053: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
    mul_395: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_1053);  mul_394 = unsqueeze_1053 = None
    unsqueeze_1054: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
    unsqueeze_1055: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
    add_337: "f32[8, 896, 7, 7]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_1055);  mul_395 = unsqueeze_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_127: "f32[8, 896, 7, 7]" = torch.ops.aten.relu.default(add_337);  add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_127 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1018: "f32[8, 112, 7, 7]" = split_with_sizes_127[0];  split_with_sizes_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_128 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1027: "f32[8, 112, 7, 7]" = split_with_sizes_128[1];  split_with_sizes_128 = None
    split_with_sizes_129 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1036: "f32[8, 112, 7, 7]" = split_with_sizes_129[2];  split_with_sizes_129 = None
    split_with_sizes_130 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1045: "f32[8, 112, 7, 7]" = split_with_sizes_130[3];  split_with_sizes_130 = None
    split_with_sizes_131 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1054: "f32[8, 112, 7, 7]" = split_with_sizes_131[4];  split_with_sizes_131 = None
    split_with_sizes_132 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1063: "f32[8, 112, 7, 7]" = split_with_sizes_132[5];  split_with_sizes_132 = None
    split_with_sizes_133 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1072: "f32[8, 112, 7, 7]" = split_with_sizes_133[6];  split_with_sizes_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_134 = torch.ops.aten.split_with_sizes.default(relu_127, [112, 112, 112, 112, 112, 112, 112, 112], 1);  relu_127 = None
    getitem_1081: "f32[8, 112, 7, 7]" = split_with_sizes_134[7];  split_with_sizes_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_132: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_1018, arg396_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1018 = arg396_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1056: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg845_1, -1);  arg845_1 = None
    unsqueeze_1057: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
    sub_132: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1057);  convolution_132 = unsqueeze_1057 = None
    add_338: "f32[112]" = torch.ops.aten.add.Tensor(arg846_1, 1e-05);  arg846_1 = None
    sqrt_132: "f32[112]" = torch.ops.aten.sqrt.default(add_338);  add_338 = None
    reciprocal_132: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
    mul_396: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
    unsqueeze_1058: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
    unsqueeze_1059: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
    mul_397: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
    unsqueeze_1060: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
    unsqueeze_1061: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
    mul_398: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_1061);  mul_397 = unsqueeze_1061 = None
    unsqueeze_1062: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
    unsqueeze_1063: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
    add_339: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_1063);  mul_398 = unsqueeze_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_128: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_339);  add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_340: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_128, getitem_1027);  getitem_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_133: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_340, arg399_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_340 = arg399_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1064: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg848_1, -1);  arg848_1 = None
    unsqueeze_1065: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
    sub_133: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1065);  convolution_133 = unsqueeze_1065 = None
    add_341: "f32[112]" = torch.ops.aten.add.Tensor(arg849_1, 1e-05);  arg849_1 = None
    sqrt_133: "f32[112]" = torch.ops.aten.sqrt.default(add_341);  add_341 = None
    reciprocal_133: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
    mul_399: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
    unsqueeze_1066: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
    unsqueeze_1067: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
    mul_400: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
    unsqueeze_1068: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
    unsqueeze_1069: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
    mul_401: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_1069);  mul_400 = unsqueeze_1069 = None
    unsqueeze_1070: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
    unsqueeze_1071: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
    add_342: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_1071);  mul_401 = unsqueeze_1071 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_129: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_342);  add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_343: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_129, getitem_1036);  getitem_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_134: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_343, arg402_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_343 = arg402_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1072: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg851_1, -1);  arg851_1 = None
    unsqueeze_1073: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
    sub_134: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_1073);  convolution_134 = unsqueeze_1073 = None
    add_344: "f32[112]" = torch.ops.aten.add.Tensor(arg852_1, 1e-05);  arg852_1 = None
    sqrt_134: "f32[112]" = torch.ops.aten.sqrt.default(add_344);  add_344 = None
    reciprocal_134: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
    mul_402: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
    unsqueeze_1074: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_402, -1);  mul_402 = None
    unsqueeze_1075: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
    mul_403: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
    unsqueeze_1076: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
    unsqueeze_1077: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
    mul_404: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_1077);  mul_403 = unsqueeze_1077 = None
    unsqueeze_1078: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
    unsqueeze_1079: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
    add_345: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_404, unsqueeze_1079);  mul_404 = unsqueeze_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_130: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_345);  add_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_346: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_130, getitem_1045);  getitem_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_135: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_346, arg405_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_346 = arg405_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1080: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg854_1, -1);  arg854_1 = None
    unsqueeze_1081: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
    sub_135: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_1081);  convolution_135 = unsqueeze_1081 = None
    add_347: "f32[112]" = torch.ops.aten.add.Tensor(arg855_1, 1e-05);  arg855_1 = None
    sqrt_135: "f32[112]" = torch.ops.aten.sqrt.default(add_347);  add_347 = None
    reciprocal_135: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
    mul_405: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
    unsqueeze_1082: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
    unsqueeze_1083: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
    mul_406: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
    unsqueeze_1084: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
    unsqueeze_1085: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
    mul_407: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_1085);  mul_406 = unsqueeze_1085 = None
    unsqueeze_1086: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
    unsqueeze_1087: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
    add_348: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_407, unsqueeze_1087);  mul_407 = unsqueeze_1087 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_131: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_348);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_349: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_131, getitem_1054);  getitem_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_136: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_349, arg408_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_349 = arg408_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1088: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg857_1, -1);  arg857_1 = None
    unsqueeze_1089: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
    sub_136: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_1089);  convolution_136 = unsqueeze_1089 = None
    add_350: "f32[112]" = torch.ops.aten.add.Tensor(arg858_1, 1e-05);  arg858_1 = None
    sqrt_136: "f32[112]" = torch.ops.aten.sqrt.default(add_350);  add_350 = None
    reciprocal_136: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
    mul_408: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
    unsqueeze_1090: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
    unsqueeze_1091: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
    mul_409: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
    unsqueeze_1092: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
    unsqueeze_1093: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
    mul_410: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_1093);  mul_409 = unsqueeze_1093 = None
    unsqueeze_1094: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
    unsqueeze_1095: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
    add_351: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_1095);  mul_410 = unsqueeze_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_132: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_351);  add_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_352: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_132, getitem_1063);  getitem_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_137: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_352, arg411_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_352 = arg411_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1096: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg860_1, -1);  arg860_1 = None
    unsqueeze_1097: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
    sub_137: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_1097);  convolution_137 = unsqueeze_1097 = None
    add_353: "f32[112]" = torch.ops.aten.add.Tensor(arg861_1, 1e-05);  arg861_1 = None
    sqrt_137: "f32[112]" = torch.ops.aten.sqrt.default(add_353);  add_353 = None
    reciprocal_137: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
    mul_411: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
    unsqueeze_1098: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
    unsqueeze_1099: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
    mul_412: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
    unsqueeze_1100: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
    unsqueeze_1101: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
    mul_413: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_1101);  mul_412 = unsqueeze_1101 = None
    unsqueeze_1102: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
    unsqueeze_1103: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
    add_354: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_1103);  mul_413 = unsqueeze_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_133: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_354);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_355: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_133, getitem_1072);  getitem_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_138: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_355, arg414_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_355 = arg414_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1104: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg863_1, -1);  arg863_1 = None
    unsqueeze_1105: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
    sub_138: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_1105);  convolution_138 = unsqueeze_1105 = None
    add_356: "f32[112]" = torch.ops.aten.add.Tensor(arg864_1, 1e-05);  arg864_1 = None
    sqrt_138: "f32[112]" = torch.ops.aten.sqrt.default(add_356);  add_356 = None
    reciprocal_138: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
    mul_414: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
    unsqueeze_1106: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
    unsqueeze_1107: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
    mul_415: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
    unsqueeze_1108: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
    unsqueeze_1109: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
    mul_416: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_1109);  mul_415 = unsqueeze_1109 = None
    unsqueeze_1110: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
    unsqueeze_1111: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
    add_357: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_1111);  mul_416 = unsqueeze_1111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_134: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_357);  add_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_14: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([relu_128, relu_129, relu_130, relu_131, relu_132, relu_133, relu_134, getitem_1081], 1);  relu_128 = relu_129 = relu_130 = relu_131 = relu_132 = relu_133 = relu_134 = getitem_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_139: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_14, arg417_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_14 = arg417_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1112: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg866_1, -1);  arg866_1 = None
    unsqueeze_1113: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
    sub_139: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_1113);  convolution_139 = unsqueeze_1113 = None
    add_358: "f32[2048]" = torch.ops.aten.add.Tensor(arg867_1, 1e-05);  arg867_1 = None
    sqrt_139: "f32[2048]" = torch.ops.aten.sqrt.default(add_358);  add_358 = None
    reciprocal_139: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
    mul_417: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
    unsqueeze_1114: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
    unsqueeze_1115: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
    mul_418: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
    unsqueeze_1116: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
    unsqueeze_1117: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
    mul_419: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_1117);  mul_418 = unsqueeze_1117 = None
    unsqueeze_1118: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
    unsqueeze_1119: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
    add_359: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_1119);  mul_419 = unsqueeze_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_360: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_359, relu_126);  add_359 = relu_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_135: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_360);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_140: "f32[8, 896, 7, 7]" = torch.ops.aten.convolution.default(relu_135, arg420_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg420_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1120: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg869_1, -1);  arg869_1 = None
    unsqueeze_1121: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
    sub_140: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_1121);  convolution_140 = unsqueeze_1121 = None
    add_361: "f32[896]" = torch.ops.aten.add.Tensor(arg870_1, 1e-05);  arg870_1 = None
    sqrt_140: "f32[896]" = torch.ops.aten.sqrt.default(add_361);  add_361 = None
    reciprocal_140: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
    mul_420: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
    unsqueeze_1122: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
    unsqueeze_1123: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
    mul_421: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
    unsqueeze_1124: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
    unsqueeze_1125: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
    mul_422: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_1125);  mul_421 = unsqueeze_1125 = None
    unsqueeze_1126: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
    unsqueeze_1127: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
    add_362: "f32[8, 896, 7, 7]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_1127);  mul_422 = unsqueeze_1127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_136: "f32[8, 896, 7, 7]" = torch.ops.aten.relu.default(add_362);  add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_136 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1090: "f32[8, 112, 7, 7]" = split_with_sizes_136[0];  split_with_sizes_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_137 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1099: "f32[8, 112, 7, 7]" = split_with_sizes_137[1];  split_with_sizes_137 = None
    split_with_sizes_138 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1108: "f32[8, 112, 7, 7]" = split_with_sizes_138[2];  split_with_sizes_138 = None
    split_with_sizes_139 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1117: "f32[8, 112, 7, 7]" = split_with_sizes_139[3];  split_with_sizes_139 = None
    split_with_sizes_140 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1126: "f32[8, 112, 7, 7]" = split_with_sizes_140[4];  split_with_sizes_140 = None
    split_with_sizes_141 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1135: "f32[8, 112, 7, 7]" = split_with_sizes_141[5];  split_with_sizes_141 = None
    split_with_sizes_142 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1)
    getitem_1144: "f32[8, 112, 7, 7]" = split_with_sizes_142[6];  split_with_sizes_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_143 = torch.ops.aten.split_with_sizes.default(relu_136, [112, 112, 112, 112, 112, 112, 112, 112], 1);  relu_136 = None
    getitem_1153: "f32[8, 112, 7, 7]" = split_with_sizes_143[7];  split_with_sizes_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_141: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_1090, arg423_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1090 = arg423_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1128: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg872_1, -1);  arg872_1 = None
    unsqueeze_1129: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
    sub_141: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_1129);  convolution_141 = unsqueeze_1129 = None
    add_363: "f32[112]" = torch.ops.aten.add.Tensor(arg873_1, 1e-05);  arg873_1 = None
    sqrt_141: "f32[112]" = torch.ops.aten.sqrt.default(add_363);  add_363 = None
    reciprocal_141: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
    mul_423: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
    unsqueeze_1130: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_423, -1);  mul_423 = None
    unsqueeze_1131: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
    mul_424: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
    unsqueeze_1132: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
    unsqueeze_1133: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
    mul_425: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_1133);  mul_424 = unsqueeze_1133 = None
    unsqueeze_1134: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
    unsqueeze_1135: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
    add_364: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_425, unsqueeze_1135);  mul_425 = unsqueeze_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_137: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_364);  add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_365: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_137, getitem_1099);  getitem_1099 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_142: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_365, arg426_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_365 = arg426_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1136: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg875_1, -1);  arg875_1 = None
    unsqueeze_1137: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
    sub_142: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_1137);  convolution_142 = unsqueeze_1137 = None
    add_366: "f32[112]" = torch.ops.aten.add.Tensor(arg876_1, 1e-05);  arg876_1 = None
    sqrt_142: "f32[112]" = torch.ops.aten.sqrt.default(add_366);  add_366 = None
    reciprocal_142: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
    mul_426: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
    unsqueeze_1138: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_426, -1);  mul_426 = None
    unsqueeze_1139: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
    mul_427: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
    unsqueeze_1140: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
    unsqueeze_1141: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
    mul_428: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_1141);  mul_427 = unsqueeze_1141 = None
    unsqueeze_1142: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
    unsqueeze_1143: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
    add_367: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_428, unsqueeze_1143);  mul_428 = unsqueeze_1143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_138: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_367);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_368: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_138, getitem_1108);  getitem_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_143: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_368, arg429_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_368 = arg429_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1144: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg878_1, -1);  arg878_1 = None
    unsqueeze_1145: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
    sub_143: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_1145);  convolution_143 = unsqueeze_1145 = None
    add_369: "f32[112]" = torch.ops.aten.add.Tensor(arg879_1, 1e-05);  arg879_1 = None
    sqrt_143: "f32[112]" = torch.ops.aten.sqrt.default(add_369);  add_369 = None
    reciprocal_143: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
    mul_429: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
    unsqueeze_1146: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
    unsqueeze_1147: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
    mul_430: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
    unsqueeze_1148: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
    unsqueeze_1149: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
    mul_431: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1149);  mul_430 = unsqueeze_1149 = None
    unsqueeze_1150: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
    unsqueeze_1151: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
    add_370: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1151);  mul_431 = unsqueeze_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_139: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_370);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_371: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_139, getitem_1117);  getitem_1117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_144: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_371, arg432_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_371 = arg432_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1152: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg881_1, -1);  arg881_1 = None
    unsqueeze_1153: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
    sub_144: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1153);  convolution_144 = unsqueeze_1153 = None
    add_372: "f32[112]" = torch.ops.aten.add.Tensor(arg882_1, 1e-05);  arg882_1 = None
    sqrt_144: "f32[112]" = torch.ops.aten.sqrt.default(add_372);  add_372 = None
    reciprocal_144: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
    mul_432: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
    unsqueeze_1154: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
    unsqueeze_1155: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
    mul_433: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
    unsqueeze_1156: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg433_1, -1);  arg433_1 = None
    unsqueeze_1157: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
    mul_434: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1157);  mul_433 = unsqueeze_1157 = None
    unsqueeze_1158: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
    unsqueeze_1159: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
    add_373: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1159);  mul_434 = unsqueeze_1159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_140: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_373);  add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_374: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_140, getitem_1126);  getitem_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_145: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_374, arg435_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_374 = arg435_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1160: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg884_1, -1);  arg884_1 = None
    unsqueeze_1161: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
    sub_145: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1161);  convolution_145 = unsqueeze_1161 = None
    add_375: "f32[112]" = torch.ops.aten.add.Tensor(arg885_1, 1e-05);  arg885_1 = None
    sqrt_145: "f32[112]" = torch.ops.aten.sqrt.default(add_375);  add_375 = None
    reciprocal_145: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
    mul_435: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
    unsqueeze_1162: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
    unsqueeze_1163: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
    mul_436: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
    unsqueeze_1164: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
    unsqueeze_1165: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
    mul_437: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1165);  mul_436 = unsqueeze_1165 = None
    unsqueeze_1166: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
    unsqueeze_1167: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
    add_376: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1167);  mul_437 = unsqueeze_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_141: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_376);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_377: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_141, getitem_1135);  getitem_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_146: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_377, arg438_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_377 = arg438_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1168: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg887_1, -1);  arg887_1 = None
    unsqueeze_1169: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
    sub_146: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_1169);  convolution_146 = unsqueeze_1169 = None
    add_378: "f32[112]" = torch.ops.aten.add.Tensor(arg888_1, 1e-05);  arg888_1 = None
    sqrt_146: "f32[112]" = torch.ops.aten.sqrt.default(add_378);  add_378 = None
    reciprocal_146: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
    mul_438: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
    unsqueeze_1170: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
    unsqueeze_1171: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
    mul_439: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
    unsqueeze_1172: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
    unsqueeze_1173: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
    mul_440: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1173);  mul_439 = unsqueeze_1173 = None
    unsqueeze_1174: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
    unsqueeze_1175: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
    add_379: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1175);  mul_440 = unsqueeze_1175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_142: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_379);  add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_380: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_142, getitem_1144);  getitem_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_147: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_380, arg441_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_380 = arg441_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1176: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg890_1, -1);  arg890_1 = None
    unsqueeze_1177: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
    sub_147: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_1177);  convolution_147 = unsqueeze_1177 = None
    add_381: "f32[112]" = torch.ops.aten.add.Tensor(arg891_1, 1e-05);  arg891_1 = None
    sqrt_147: "f32[112]" = torch.ops.aten.sqrt.default(add_381);  add_381 = None
    reciprocal_147: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
    mul_441: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
    unsqueeze_1178: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
    unsqueeze_1179: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
    mul_442: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
    unsqueeze_1180: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
    unsqueeze_1181: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
    mul_443: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1181);  mul_442 = unsqueeze_1181 = None
    unsqueeze_1182: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
    unsqueeze_1183: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
    add_382: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1183);  mul_443 = unsqueeze_1183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_143: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_382);  add_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_15: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([relu_137, relu_138, relu_139, relu_140, relu_141, relu_142, relu_143, getitem_1153], 1);  relu_137 = relu_138 = relu_139 = relu_140 = relu_141 = relu_142 = relu_143 = getitem_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_148: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_15, arg444_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_15 = arg444_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1184: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg893_1, -1);  arg893_1 = None
    unsqueeze_1185: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
    sub_148: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_1185);  convolution_148 = unsqueeze_1185 = None
    add_383: "f32[2048]" = torch.ops.aten.add.Tensor(arg894_1, 1e-05);  arg894_1 = None
    sqrt_148: "f32[2048]" = torch.ops.aten.sqrt.default(add_383);  add_383 = None
    reciprocal_148: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
    mul_444: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
    unsqueeze_1186: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
    unsqueeze_1187: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
    mul_445: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
    unsqueeze_1188: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
    unsqueeze_1189: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
    mul_446: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1189);  mul_445 = unsqueeze_1189 = None
    unsqueeze_1190: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
    unsqueeze_1191: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
    add_384: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1191);  mul_446 = unsqueeze_1191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_385: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_384, relu_135);  add_384 = relu_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_144: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_385);  add_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_144, [-1, -2], True);  relu_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 2048]" = torch.ops.aten.reshape.default(mean, [8, 2048]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg448_1, view, permute);  arg448_1 = view = permute = None
    return (addmm,)
    