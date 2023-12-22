from __future__ import annotations



def forward(self, arg0_1: "f32[768]", arg1_1: "f32[16, 1, 1]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[16, 1, 1]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[16, 1, 1]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[16, 1, 1]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[16, 1, 1]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[16, 1, 1]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[16, 1, 1]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[16, 1, 1]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[16, 1, 1]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[16, 1, 1]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[16, 1, 1]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[16, 1, 1]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[16, 1, 1]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[16, 1, 1]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[16, 1, 1]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[16, 1, 1]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[16, 1, 1]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[16, 1, 1]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[16, 1, 1]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[16, 1, 1]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[768]", arg81_1: "f32[16, 1, 1]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[16, 1, 1]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[16, 1, 1]", arg90_1: "f32[768]", arg91_1: "f32[768]", arg92_1: "f32[768]", arg93_1: "f32[16, 1, 1]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[1, 1, 768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[192, 3, 3, 3]", arg102_1: "f32[192]", arg103_1: "f32[192]", arg104_1: "f32[384, 192, 3, 3]", arg105_1: "f32[384]", arg106_1: "f32[384]", arg107_1: "f32[768, 384, 3, 3]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[768, 64, 1, 1]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[2304, 768]", arg115_1: "f32[2304]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768, 1, 3, 3]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768, 1, 3, 3]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[3072, 768]", arg129_1: "f32[3072]", arg130_1: "f32[768, 3072]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[768]", arg134_1: "f32[2304, 768]", arg135_1: "f32[2304]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768, 1, 3, 3]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768, 1, 3, 3]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[3072, 768]", arg149_1: "f32[3072]", arg150_1: "f32[768, 3072]", arg151_1: "f32[768]", arg152_1: "f32[768]", arg153_1: "f32[768]", arg154_1: "f32[2304, 768]", arg155_1: "f32[2304]", arg156_1: "f32[768, 768]", arg157_1: "f32[768]", arg158_1: "f32[768]", arg159_1: "f32[768]", arg160_1: "f32[768, 1, 3, 3]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[768, 1, 3, 3]", arg165_1: "f32[768]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[3072, 768]", arg169_1: "f32[3072]", arg170_1: "f32[768, 3072]", arg171_1: "f32[768]", arg172_1: "f32[768]", arg173_1: "f32[768]", arg174_1: "f32[2304, 768]", arg175_1: "f32[2304]", arg176_1: "f32[768, 768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "f32[768, 1, 3, 3]", arg181_1: "f32[768]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg184_1: "f32[768, 1, 3, 3]", arg185_1: "f32[768]", arg186_1: "f32[768]", arg187_1: "f32[768]", arg188_1: "f32[3072, 768]", arg189_1: "f32[3072]", arg190_1: "f32[768, 3072]", arg191_1: "f32[768]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[2304, 768]", arg195_1: "f32[2304]", arg196_1: "f32[768, 768]", arg197_1: "f32[768]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[768, 1, 3, 3]", arg201_1: "f32[768]", arg202_1: "f32[768]", arg203_1: "f32[768]", arg204_1: "f32[768, 1, 3, 3]", arg205_1: "f32[768]", arg206_1: "f32[768]", arg207_1: "f32[768]", arg208_1: "f32[3072, 768]", arg209_1: "f32[3072]", arg210_1: "f32[768, 3072]", arg211_1: "f32[768]", arg212_1: "f32[768]", arg213_1: "f32[768]", arg214_1: "f32[2304, 768]", arg215_1: "f32[2304]", arg216_1: "f32[768, 768]", arg217_1: "f32[768]", arg218_1: "f32[768]", arg219_1: "f32[768]", arg220_1: "f32[768, 1, 3, 3]", arg221_1: "f32[768]", arg222_1: "f32[768]", arg223_1: "f32[768]", arg224_1: "f32[768, 1, 3, 3]", arg225_1: "f32[768]", arg226_1: "f32[768]", arg227_1: "f32[768]", arg228_1: "f32[3072, 768]", arg229_1: "f32[3072]", arg230_1: "f32[768, 3072]", arg231_1: "f32[768]", arg232_1: "f32[768]", arg233_1: "f32[768]", arg234_1: "f32[2304, 768]", arg235_1: "f32[2304]", arg236_1: "f32[768, 768]", arg237_1: "f32[768]", arg238_1: "f32[768]", arg239_1: "f32[768]", arg240_1: "f32[768, 1, 3, 3]", arg241_1: "f32[768]", arg242_1: "f32[768]", arg243_1: "f32[768]", arg244_1: "f32[768, 1, 3, 3]", arg245_1: "f32[768]", arg246_1: "f32[768]", arg247_1: "f32[768]", arg248_1: "f32[3072, 768]", arg249_1: "f32[3072]", arg250_1: "f32[768, 3072]", arg251_1: "f32[768]", arg252_1: "f32[768]", arg253_1: "f32[768]", arg254_1: "f32[2304, 768]", arg255_1: "f32[2304]", arg256_1: "f32[768, 768]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[768]", arg260_1: "f32[768, 1, 3, 3]", arg261_1: "f32[768]", arg262_1: "f32[768]", arg263_1: "f32[768]", arg264_1: "f32[768, 1, 3, 3]", arg265_1: "f32[768]", arg266_1: "f32[768]", arg267_1: "f32[768]", arg268_1: "f32[3072, 768]", arg269_1: "f32[3072]", arg270_1: "f32[768, 3072]", arg271_1: "f32[768]", arg272_1: "f32[768]", arg273_1: "f32[768]", arg274_1: "f32[2304, 768]", arg275_1: "f32[2304]", arg276_1: "f32[768, 768]", arg277_1: "f32[768]", arg278_1: "f32[768]", arg279_1: "f32[768]", arg280_1: "f32[768, 1, 3, 3]", arg281_1: "f32[768]", arg282_1: "f32[768]", arg283_1: "f32[768]", arg284_1: "f32[768, 1, 3, 3]", arg285_1: "f32[768]", arg286_1: "f32[768]", arg287_1: "f32[768]", arg288_1: "f32[3072, 768]", arg289_1: "f32[3072]", arg290_1: "f32[768, 3072]", arg291_1: "f32[768]", arg292_1: "f32[768]", arg293_1: "f32[768]", arg294_1: "f32[2304, 768]", arg295_1: "f32[2304]", arg296_1: "f32[768, 768]", arg297_1: "f32[768]", arg298_1: "f32[768]", arg299_1: "f32[768]", arg300_1: "f32[768, 1, 3, 3]", arg301_1: "f32[768]", arg302_1: "f32[768]", arg303_1: "f32[768]", arg304_1: "f32[768, 1, 3, 3]", arg305_1: "f32[768]", arg306_1: "f32[768]", arg307_1: "f32[768]", arg308_1: "f32[3072, 768]", arg309_1: "f32[3072]", arg310_1: "f32[768, 3072]", arg311_1: "f32[768]", arg312_1: "f32[768]", arg313_1: "f32[768]", arg314_1: "f32[2304, 768]", arg315_1: "f32[2304]", arg316_1: "f32[768, 768]", arg317_1: "f32[768]", arg318_1: "f32[768]", arg319_1: "f32[768]", arg320_1: "f32[768, 1, 3, 3]", arg321_1: "f32[768]", arg322_1: "f32[768]", arg323_1: "f32[768]", arg324_1: "f32[768, 1, 3, 3]", arg325_1: "f32[768]", arg326_1: "f32[768]", arg327_1: "f32[768]", arg328_1: "f32[3072, 768]", arg329_1: "f32[3072]", arg330_1: "f32[768, 3072]", arg331_1: "f32[768]", arg332_1: "f32[768]", arg333_1: "f32[768]", arg334_1: "f32[2304, 768]", arg335_1: "f32[2304]", arg336_1: "f32[768, 768]", arg337_1: "f32[768]", arg338_1: "f32[768]", arg339_1: "f32[768]", arg340_1: "f32[768, 1, 3, 3]", arg341_1: "f32[768]", arg342_1: "f32[768]", arg343_1: "f32[768]", arg344_1: "f32[768, 1, 3, 3]", arg345_1: "f32[768]", arg346_1: "f32[768]", arg347_1: "f32[768]", arg348_1: "f32[3072, 768]", arg349_1: "f32[3072]", arg350_1: "f32[768, 3072]", arg351_1: "f32[768]", arg352_1: "f32[768]", arg353_1: "f32[768]", arg354_1: "f32[2304, 768]", arg355_1: "f32[2304]", arg356_1: "f32[768, 768]", arg357_1: "f32[768]", arg358_1: "f32[768]", arg359_1: "f32[768]", arg360_1: "f32[768, 1, 3, 3]", arg361_1: "f32[768]", arg362_1: "f32[768]", arg363_1: "f32[768]", arg364_1: "f32[768, 1, 3, 3]", arg365_1: "f32[768]", arg366_1: "f32[768]", arg367_1: "f32[768]", arg368_1: "f32[3072, 768]", arg369_1: "f32[3072]", arg370_1: "f32[768, 3072]", arg371_1: "f32[768]", arg372_1: "f32[768]", arg373_1: "f32[768]", arg374_1: "f32[2304, 768]", arg375_1: "f32[2304]", arg376_1: "f32[768, 768]", arg377_1: "f32[768]", arg378_1: "f32[768]", arg379_1: "f32[768]", arg380_1: "f32[768, 1, 3, 3]", arg381_1: "f32[768]", arg382_1: "f32[768]", arg383_1: "f32[768]", arg384_1: "f32[768, 1, 3, 3]", arg385_1: "f32[768]", arg386_1: "f32[768]", arg387_1: "f32[768]", arg388_1: "f32[3072, 768]", arg389_1: "f32[3072]", arg390_1: "f32[768, 3072]", arg391_1: "f32[768]", arg392_1: "f32[768]", arg393_1: "f32[768]", arg394_1: "f32[2304, 768]", arg395_1: "f32[2304]", arg396_1: "f32[768, 768]", arg397_1: "f32[768]", arg398_1: "f32[768]", arg399_1: "f32[768]", arg400_1: "f32[768, 1, 3, 3]", arg401_1: "f32[768]", arg402_1: "f32[768]", arg403_1: "f32[768]", arg404_1: "f32[768, 1, 3, 3]", arg405_1: "f32[768]", arg406_1: "f32[768]", arg407_1: "f32[768]", arg408_1: "f32[3072, 768]", arg409_1: "f32[3072]", arg410_1: "f32[768, 3072]", arg411_1: "f32[768]", arg412_1: "f32[768]", arg413_1: "f32[768]", arg414_1: "f32[2304, 768]", arg415_1: "f32[2304]", arg416_1: "f32[768, 768]", arg417_1: "f32[768]", arg418_1: "f32[768]", arg419_1: "f32[768]", arg420_1: "f32[768, 1, 3, 3]", arg421_1: "f32[768]", arg422_1: "f32[768]", arg423_1: "f32[768]", arg424_1: "f32[768, 1, 3, 3]", arg425_1: "f32[768]", arg426_1: "f32[768]", arg427_1: "f32[768]", arg428_1: "f32[3072, 768]", arg429_1: "f32[3072]", arg430_1: "f32[768, 3072]", arg431_1: "f32[768]", arg432_1: "f32[768]", arg433_1: "f32[768]", arg434_1: "f32[2304, 768]", arg435_1: "f32[2304]", arg436_1: "f32[768, 768]", arg437_1: "f32[768]", arg438_1: "f32[768]", arg439_1: "f32[768]", arg440_1: "f32[768, 1, 3, 3]", arg441_1: "f32[768]", arg442_1: "f32[768]", arg443_1: "f32[768]", arg444_1: "f32[768, 1, 3, 3]", arg445_1: "f32[768]", arg446_1: "f32[768]", arg447_1: "f32[768]", arg448_1: "f32[3072, 768]", arg449_1: "f32[3072]", arg450_1: "f32[768, 3072]", arg451_1: "f32[768]", arg452_1: "f32[768]", arg453_1: "f32[768]", arg454_1: "f32[2304, 768]", arg455_1: "f32[2304]", arg456_1: "f32[768, 768]", arg457_1: "f32[768]", arg458_1: "f32[768]", arg459_1: "f32[768]", arg460_1: "f32[768, 1, 3, 3]", arg461_1: "f32[768]", arg462_1: "f32[768]", arg463_1: "f32[768]", arg464_1: "f32[768, 1, 3, 3]", arg465_1: "f32[768]", arg466_1: "f32[768]", arg467_1: "f32[768]", arg468_1: "f32[3072, 768]", arg469_1: "f32[3072]", arg470_1: "f32[768, 3072]", arg471_1: "f32[768]", arg472_1: "f32[768]", arg473_1: "f32[768]", arg474_1: "f32[2304, 768]", arg475_1: "f32[2304]", arg476_1: "f32[768, 768]", arg477_1: "f32[768]", arg478_1: "f32[768]", arg479_1: "f32[768]", arg480_1: "f32[768, 1, 3, 3]", arg481_1: "f32[768]", arg482_1: "f32[768]", arg483_1: "f32[768]", arg484_1: "f32[768, 1, 3, 3]", arg485_1: "f32[768]", arg486_1: "f32[768]", arg487_1: "f32[768]", arg488_1: "f32[3072, 768]", arg489_1: "f32[3072]", arg490_1: "f32[768, 3072]", arg491_1: "f32[768]", arg492_1: "f32[768]", arg493_1: "f32[768]", arg494_1: "f32[2304, 768]", arg495_1: "f32[2304]", arg496_1: "f32[768, 768]", arg497_1: "f32[768]", arg498_1: "f32[768]", arg499_1: "f32[768]", arg500_1: "f32[768, 1, 3, 3]", arg501_1: "f32[768]", arg502_1: "f32[768]", arg503_1: "f32[768]", arg504_1: "f32[768, 1, 3, 3]", arg505_1: "f32[768]", arg506_1: "f32[768]", arg507_1: "f32[768]", arg508_1: "f32[3072, 768]", arg509_1: "f32[3072]", arg510_1: "f32[768, 3072]", arg511_1: "f32[768]", arg512_1: "f32[768]", arg513_1: "f32[768]", arg514_1: "f32[2304, 768]", arg515_1: "f32[2304]", arg516_1: "f32[768, 768]", arg517_1: "f32[768]", arg518_1: "f32[768]", arg519_1: "f32[768]", arg520_1: "f32[768, 1, 3, 3]", arg521_1: "f32[768]", arg522_1: "f32[768]", arg523_1: "f32[768]", arg524_1: "f32[768, 1, 3, 3]", arg525_1: "f32[768]", arg526_1: "f32[768]", arg527_1: "f32[768]", arg528_1: "f32[3072, 768]", arg529_1: "f32[3072]", arg530_1: "f32[768, 3072]", arg531_1: "f32[768]", arg532_1: "f32[768]", arg533_1: "f32[768]", arg534_1: "f32[2304, 768]", arg535_1: "f32[2304]", arg536_1: "f32[768, 768]", arg537_1: "f32[768]", arg538_1: "f32[768]", arg539_1: "f32[768]", arg540_1: "f32[768, 1, 3, 3]", arg541_1: "f32[768]", arg542_1: "f32[768]", arg543_1: "f32[768]", arg544_1: "f32[768, 1, 3, 3]", arg545_1: "f32[768]", arg546_1: "f32[768]", arg547_1: "f32[768]", arg548_1: "f32[3072, 768]", arg549_1: "f32[3072]", arg550_1: "f32[768, 3072]", arg551_1: "f32[768]", arg552_1: "f32[768]", arg553_1: "f32[768]", arg554_1: "f32[2304, 768]", arg555_1: "f32[2304]", arg556_1: "f32[768, 768]", arg557_1: "f32[768]", arg558_1: "f32[768]", arg559_1: "f32[768]", arg560_1: "f32[768, 1, 3, 3]", arg561_1: "f32[768]", arg562_1: "f32[768]", arg563_1: "f32[768]", arg564_1: "f32[768, 1, 3, 3]", arg565_1: "f32[768]", arg566_1: "f32[768]", arg567_1: "f32[768]", arg568_1: "f32[3072, 768]", arg569_1: "f32[3072]", arg570_1: "f32[768, 3072]", arg571_1: "f32[768]", arg572_1: "f32[768]", arg573_1: "f32[768]", arg574_1: "f32[2304, 768]", arg575_1: "f32[2304]", arg576_1: "f32[768, 768]", arg577_1: "f32[768]", arg578_1: "f32[768]", arg579_1: "f32[768]", arg580_1: "f32[768, 1, 3, 3]", arg581_1: "f32[768]", arg582_1: "f32[768]", arg583_1: "f32[768]", arg584_1: "f32[768, 1, 3, 3]", arg585_1: "f32[768]", arg586_1: "f32[768]", arg587_1: "f32[768]", arg588_1: "f32[3072, 768]", arg589_1: "f32[3072]", arg590_1: "f32[768, 3072]", arg591_1: "f32[768]", arg592_1: "f32[768]", arg593_1: "f32[768]", arg594_1: "f32[768, 768]", arg595_1: "f32[768]", arg596_1: "f32[768, 768]", arg597_1: "f32[768]", arg598_1: "f32[768, 768]", arg599_1: "f32[768]", arg600_1: "f32[768, 768]", arg601_1: "f32[768]", arg602_1: "f32[768]", arg603_1: "f32[768]", arg604_1: "f32[3072, 768]", arg605_1: "f32[3072]", arg606_1: "f32[768, 3072]", arg607_1: "f32[768]", arg608_1: "f32[768]", arg609_1: "f32[768]", arg610_1: "f32[768, 768]", arg611_1: "f32[768]", arg612_1: "f32[768, 768]", arg613_1: "f32[768]", arg614_1: "f32[768, 768]", arg615_1: "f32[768]", arg616_1: "f32[768, 768]", arg617_1: "f32[768]", arg618_1: "f32[768]", arg619_1: "f32[768]", arg620_1: "f32[3072, 768]", arg621_1: "f32[3072]", arg622_1: "f32[768, 3072]", arg623_1: "f32[768]", arg624_1: "f32[768]", arg625_1: "f32[768]", arg626_1: "f32[1000, 768]", arg627_1: "f32[1000]", arg628_1: "f32[192]", arg629_1: "f32[192]", arg630_1: "i64[]", arg631_1: "f32[384]", arg632_1: "f32[384]", arg633_1: "i64[]", arg634_1: "f32[768]", arg635_1: "f32[768]", arg636_1: "i64[]", arg637_1: "f32[768]", arg638_1: "f32[768]", arg639_1: "i64[]", arg640_1: "f32[768]", arg641_1: "f32[768]", arg642_1: "i64[]", arg643_1: "f32[768]", arg644_1: "f32[768]", arg645_1: "i64[]", arg646_1: "f32[768]", arg647_1: "f32[768]", arg648_1: "i64[]", arg649_1: "f32[768]", arg650_1: "f32[768]", arg651_1: "i64[]", arg652_1: "f32[768]", arg653_1: "f32[768]", arg654_1: "i64[]", arg655_1: "f32[768]", arg656_1: "f32[768]", arg657_1: "i64[]", arg658_1: "f32[768]", arg659_1: "f32[768]", arg660_1: "i64[]", arg661_1: "f32[768]", arg662_1: "f32[768]", arg663_1: "i64[]", arg664_1: "f32[768]", arg665_1: "f32[768]", arg666_1: "i64[]", arg667_1: "f32[768]", arg668_1: "f32[768]", arg669_1: "i64[]", arg670_1: "f32[768]", arg671_1: "f32[768]", arg672_1: "i64[]", arg673_1: "f32[768]", arg674_1: "f32[768]", arg675_1: "i64[]", arg676_1: "f32[768]", arg677_1: "f32[768]", arg678_1: "i64[]", arg679_1: "f32[768]", arg680_1: "f32[768]", arg681_1: "i64[]", arg682_1: "f32[768]", arg683_1: "f32[768]", arg684_1: "i64[]", arg685_1: "f32[768]", arg686_1: "f32[768]", arg687_1: "i64[]", arg688_1: "f32[768]", arg689_1: "f32[768]", arg690_1: "i64[]", arg691_1: "f32[768]", arg692_1: "f32[768]", arg693_1: "i64[]", arg694_1: "f32[768]", arg695_1: "f32[768]", arg696_1: "i64[]", arg697_1: "f32[768]", arg698_1: "f32[768]", arg699_1: "i64[]", arg700_1: "f32[768]", arg701_1: "f32[768]", arg702_1: "i64[]", arg703_1: "f32[768]", arg704_1: "f32[768]", arg705_1: "i64[]", arg706_1: "f32[768]", arg707_1: "f32[768]", arg708_1: "i64[]", arg709_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:107, code: x = self.proj(x)
    convolution: "f32[8, 192, 112, 112]" = torch.ops.aten.convolution.default(arg709_1, arg101_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg709_1 = arg101_1 = None
    convert_element_type: "f32[192]" = torch.ops.prims.convert_element_type.default(arg628_1, torch.float32);  arg628_1 = None
    convert_element_type_1: "f32[192]" = torch.ops.prims.convert_element_type.default(arg629_1, torch.float32);  arg629_1 = None
    add: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[192]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
    unsqueeze_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    mul_3: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, 0.5)
    mul_4: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, 0.7071067811865476);  add_1 = None
    erf: "f32[8, 192, 112, 112]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_2: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_5: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_3, add_2);  mul_3 = add_2 = None
    convolution_1: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(mul_5, arg104_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_5 = arg104_1 = None
    convert_element_type_2: "f32[384]" = torch.ops.prims.convert_element_type.default(arg631_1, torch.float32);  arg631_1 = None
    convert_element_type_3: "f32[384]" = torch.ops.prims.convert_element_type.default(arg632_1, torch.float32);  arg632_1 = None
    add_3: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[384]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
    reciprocal_1: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_6: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 384, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_7: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
    unsqueeze_13: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_8: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_13);  mul_7 = unsqueeze_13 = None
    unsqueeze_14: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_15: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_4: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_15);  mul_8 = unsqueeze_15 = None
    mul_9: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_4, 0.5)
    mul_10: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_4, 0.7071067811865476);  add_4 = None
    erf_1: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
    add_5: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_11: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_9, add_5);  mul_9 = add_5 = None
    convolution_2: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(mul_11, arg107_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_11 = arg107_1 = None
    convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(arg634_1, torch.float32);  arg634_1 = None
    convert_element_type_5: "f32[768]" = torch.ops.prims.convert_element_type.default(arg635_1, torch.float32);  arg635_1 = None
    add_6: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[768]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_2: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_12: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_19: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_13: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
    unsqueeze_21: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_14: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_21);  mul_13 = unsqueeze_21 = None
    unsqueeze_22: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_23: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_7: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_23);  mul_14 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:109, code: x = x.flatten(2).transpose(1, 2)  # (B, N, C)
    view: "f32[8, 768, 784]" = torch.ops.aten.view.default(add_7, [8, 768, 784]);  add_7 = None
    permute: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:51, code: y_embed = torch.arange(1, H+1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
    iota: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    convert_element_type_6: "f64[28]" = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
    mul_15: "f64[28]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1);  convert_element_type_6 = None
    add_8: "f64[28]" = torch.ops.aten.add.Tensor(mul_15, 1);  mul_15 = None
    convert_element_type_7: "f32[28]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
    unsqueeze_24: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_7, 1);  convert_element_type_7 = None
    repeat: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(unsqueeze_24, [1, 1, 28]);  unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:52, code: x_embed = torch.arange(1, W+1, dtype=torch.float32, device=device).repeat(1, H, 1)
    iota_1: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    convert_element_type_8: "f64[28]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
    mul_16: "f64[28]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1);  convert_element_type_8 = None
    add_9: "f64[28]" = torch.ops.aten.add.Tensor(mul_16, 1);  mul_16 = None
    convert_element_type_9: "f32[28]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
    repeat_1: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(convert_element_type_9, [1, 28, 1]);  convert_element_type_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:53, code: y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
    slice_1: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(repeat, 0, 0, 9223372036854775807)
    slice_2: "f32[1, 1, 28]" = torch.ops.aten.slice.Tensor(slice_1, 1, -1, 9223372036854775807);  slice_1 = None
    slice_3: "f32[1, 1, 28]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 9223372036854775807);  slice_2 = None
    add_10: "f32[1, 1, 28]" = torch.ops.aten.add.Tensor(slice_3, 1e-06);  slice_3 = None
    div: "f32[1, 28, 28]" = torch.ops.aten.div.Tensor(repeat, add_10);  repeat = add_10 = None
    mul_17: "f32[1, 28, 28]" = torch.ops.aten.mul.Tensor(div, 6.283185307179586);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:54, code: x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
    slice_4: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(repeat_1, 0, 0, 9223372036854775807)
    slice_5: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 9223372036854775807);  slice_4 = None
    slice_6: "f32[1, 28, 1]" = torch.ops.aten.slice.Tensor(slice_5, 2, -1, 9223372036854775807);  slice_5 = None
    add_11: "f32[1, 28, 1]" = torch.ops.aten.add.Tensor(slice_6, 1e-06);  slice_6 = None
    div_1: "f32[1, 28, 28]" = torch.ops.aten.div.Tensor(repeat_1, add_11);  repeat_1 = add_11 = None
    mul_18: "f32[1, 28, 28]" = torch.ops.aten.mul.Tensor(div_1, 6.283185307179586);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:55, code: dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
    iota_2: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    convert_element_type_10: "f64[32]" = torch.ops.prims.convert_element_type.default(iota_2, torch.float64);  iota_2 = None
    mul_19: "f64[32]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1);  convert_element_type_10 = None
    add_12: "f64[32]" = torch.ops.aten.add.Tensor(mul_19, 0);  mul_19 = None
    convert_element_type_11: "f32[32]" = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:56, code: dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
    div_2: "f32[32]" = torch.ops.aten.div.Tensor_mode(convert_element_type_11, 2, rounding_mode = 'floor');  convert_element_type_11 = None
    mul_20: "f32[32]" = torch.ops.aten.mul.Tensor(div_2, 2);  div_2 = None
    div_3: "f32[32]" = torch.ops.aten.div.Tensor(mul_20, 32);  mul_20 = None
    pow_1: "f32[32]" = torch.ops.aten.pow.Scalar(10000, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:57, code: pos_x = x_embed[:, :, :, None] / dim_t
    slice_7: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(mul_18, 0, 0, 9223372036854775807);  mul_18 = None
    slice_8: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    slice_9: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_8, 2, 0, 9223372036854775807);  slice_8 = None
    unsqueeze_25: "f32[1, 28, 28, 1]" = torch.ops.aten.unsqueeze.default(slice_9, 3);  slice_9 = None
    div_4: "f32[1, 28, 28, 32]" = torch.ops.aten.div.Tensor(unsqueeze_25, pow_1);  unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:58, code: pos_y = y_embed[:, :, :, None] / dim_t
    slice_10: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(mul_17, 0, 0, 9223372036854775807);  mul_17 = None
    slice_11: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 9223372036854775807);  slice_10 = None
    slice_12: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_11, 2, 0, 9223372036854775807);  slice_11 = None
    unsqueeze_26: "f32[1, 28, 28, 1]" = torch.ops.aten.unsqueeze.default(slice_12, 3);  slice_12 = None
    div_5: "f32[1, 28, 28, 32]" = torch.ops.aten.div.Tensor(unsqueeze_26, pow_1);  unsqueeze_26 = pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:59, code: pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
    slice_13: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_4, 0, 0, 9223372036854775807)
    slice_14: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807);  slice_14 = None
    slice_16: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 9223372036854775807, 2);  slice_15 = None
    sin: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_16);  slice_16 = None
    slice_17: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_4, 0, 0, 9223372036854775807);  div_4 = None
    slice_18: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_18, 2, 0, 9223372036854775807);  slice_18 = None
    slice_20: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_19, 3, 1, 9223372036854775807, 2);  slice_19 = None
    cos: "f32[1, 28, 28, 16]" = torch.ops.aten.cos.default(slice_20);  slice_20 = None
    unsqueeze_27: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(sin, 4);  sin = None
    unsqueeze_28: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(cos, 4);  cos = None
    cat: "f32[1, 28, 28, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_27, unsqueeze_28], 4);  unsqueeze_27 = unsqueeze_28 = None
    view_1: "f32[1, 28, 28, 32]" = torch.ops.aten.view.default(cat, [1, 28, 28, 32]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:60, code: pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
    slice_21: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_5, 0, 0, 9223372036854775807)
    slice_22: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 9223372036854775807);  slice_22 = None
    slice_24: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 9223372036854775807, 2);  slice_23 = None
    sin_1: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_24);  slice_24 = None
    slice_25: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_5, 0, 0, 9223372036854775807);  div_5 = None
    slice_26: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    slice_27: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 9223372036854775807);  slice_26 = None
    slice_28: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_27, 3, 1, 9223372036854775807, 2);  slice_27 = None
    cos_1: "f32[1, 28, 28, 16]" = torch.ops.aten.cos.default(slice_28);  slice_28 = None
    unsqueeze_29: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(sin_1, 4);  sin_1 = None
    unsqueeze_30: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(cos_1, 4);  cos_1 = None
    cat_1: "f32[1, 28, 28, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_29, unsqueeze_30], 4);  unsqueeze_29 = unsqueeze_30 = None
    view_2: "f32[1, 28, 28, 32]" = torch.ops.aten.view.default(cat_1, [1, 28, 28, 32]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:61, code: pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    cat_2: "f32[1, 28, 28, 64]" = torch.ops.aten.cat.default([view_2, view_1], 3);  view_2 = view_1 = None
    permute_1: "f32[1, 64, 28, 28]" = torch.ops.aten.permute.default(cat_2, [0, 3, 1, 2]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:62, code: pos = self.token_projection(pos)
    convolution_3: "f32[1, 768, 28, 28]" = torch.ops.aten.convolution.default(permute_1, arg110_1, arg111_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_1 = arg110_1 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:63, code: return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)
    repeat_2: "f32[8, 768, 28, 28]" = torch.ops.aten.repeat.default(convolution_3, [8, 1, 1, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:437, code: pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
    view_3: "f32[8, 768, 784]" = torch.ops.aten.view.default(repeat_2, [8, -1, 784]);  repeat_2 = None
    permute_2: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:438, code: x = x + pos_encoding
    add_13: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(permute, permute_2);  permute = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:439, code: x = self.pos_drop(x)
    clone: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_1: "f32[8, 784, 768]" = torch.ops.aten.clone.default(clone, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 784, 1]" = var_mean[0]
    getitem_1: "f32[8, 784, 1]" = var_mean[1];  var_mean = None
    add_14: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_3: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
    mul_21: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
    mul_22: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg112_1);  mul_21 = arg112_1 = None
    add_15: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_22, arg113_1);  mul_22 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_4: "f32[6272, 768]" = torch.ops.aten.view.default(add_15, [6272, 768]);  add_15 = None
    permute_3: "f32[768, 2304]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg115_1, view_4, permute_3);  arg115_1 = view_4 = permute_3 = None
    view_5: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm, [8, 784, 2304]);  addmm = None
    view_6: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_5, [8, 784, 3, 16, 48]);  view_5 = None
    permute_4: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_6, [2, 0, 3, 4, 1]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_4);  permute_4 = None
    getitem_2: "f32[8, 16, 48, 784]" = unbind[0]
    getitem_3: "f32[8, 16, 48, 784]" = unbind[1]
    getitem_4: "f32[8, 16, 48, 784]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_2: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_2, 2.0)
    sum_1: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_2, [-1], True);  pow_2 = None
    pow_3: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
    clamp_min: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_3, 1e-12);  pow_3 = None
    expand: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min, [8, 16, 48, 784]);  clamp_min = None
    div_6: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_2, expand);  getitem_2 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_4: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_3, 2.0)
    sum_2: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_4, [-1], True);  pow_4 = None
    pow_5: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
    clamp_min_1: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_5, 1e-12);  pow_5 = None
    expand_1: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_1, [8, 16, 48, 784]);  clamp_min_1 = None
    div_7: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_3, expand_1);  getitem_3 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_5: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_7, [0, 1, 3, 2]);  div_7 = None
    expand_2: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_6, [8, 16, 48, 784]);  div_6 = None
    clone_2: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_7: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_2, [128, 48, 784]);  clone_2 = None
    expand_3: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_5, [8, 16, 784, 48]);  permute_5 = None
    clone_3: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_8: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_3, [128, 784, 48]);  clone_3 = None
    bmm: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_7, view_8);  view_7 = view_8 = None
    view_9: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm, [8, 16, 48, 48]);  bmm = None
    mul_23: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_9, arg1_1);  view_9 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_23, [-1], True)
    sub_4: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_23, amax);  mul_23 = amax = None
    exp: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_3: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_8: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp, sum_3);  exp = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_4: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_4: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_4, [8, 16, 48, 48]);  clone_4 = None
    view_10: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_4, [128, 48, 48]);  expand_4 = None
    expand_5: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_4, [8, 16, 48, 784]);  getitem_4 = None
    clone_5: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_11: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_5, [128, 48, 784]);  clone_5 = None
    bmm_1: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_1, [8, 16, 48, 784]);  bmm_1 = None
    permute_6: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_12, [0, 3, 1, 2]);  view_12 = None
    view_13: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_6, [8, 784, 768]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    clone_6: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_13, memory_format = torch.contiguous_format);  view_13 = None
    view_14: "f32[6272, 768]" = torch.ops.aten.view.default(clone_6, [6272, 768]);  clone_6 = None
    mm: "f32[6272, 768]" = torch.ops.aten.mm.default(view_14, permute_7);  view_14 = permute_7 = None
    view_15: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm, [8, 784, 768]);  mm = None
    add_16: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_15, arg117_1);  view_15 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_7: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_24: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg0_1, clone_7);  arg0_1 = clone_7 = None
    add_17: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(clone, mul_24);  clone = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_8: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_8, [2], correction = 0, keepdim = True)
    getitem_5: "f32[8, 784, 1]" = var_mean_1[0]
    getitem_6: "f32[8, 784, 1]" = var_mean_1[1];  var_mean_1 = None
    add_18: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
    rsqrt_1: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_8, getitem_6);  clone_8 = getitem_6 = None
    mul_25: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
    mul_26: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg118_1);  mul_25 = arg118_1 = None
    add_19: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_26, arg119_1);  mul_26 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_8: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_19, [0, 2, 1]);  add_19 = None
    view_16: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_8, [8, 768, 28, 28]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_4: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_16, arg120_1, arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_16 = arg120_1 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_27: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, 0.5)
    mul_28: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, 0.7071067811865476);  convolution_4 = None
    erf_2: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_20: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_29: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_27, add_20);  mul_27 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_12: "f32[768]" = torch.ops.prims.convert_element_type.default(arg637_1, torch.float32);  arg637_1 = None
    convert_element_type_13: "f32[768]" = torch.ops.prims.convert_element_type.default(arg638_1, torch.float32);  arg638_1 = None
    add_21: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_3: "f32[768]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_3: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_30: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_31: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_32: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, -1);  unsqueeze_31 = None
    unsqueeze_33: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_34: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, -1);  unsqueeze_33 = None
    sub_6: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_29, unsqueeze_32);  mul_29 = unsqueeze_32 = None
    mul_31: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_34);  sub_6 = unsqueeze_34 = None
    unsqueeze_35: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_36: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, -1);  unsqueeze_35 = None
    mul_32: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_36);  mul_31 = unsqueeze_36 = None
    unsqueeze_37: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_38: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, -1);  unsqueeze_37 = None
    add_22: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_38);  mul_32 = unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_5: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_22, arg124_1, arg125_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_22 = arg124_1 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_17: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_5, [8, 768, 784]);  convolution_5 = None
    permute_9: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_33: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg2_1, permute_9);  arg2_1 = permute_9 = None
    add_23: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_17, mul_33);  add_17 = mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_9: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_23, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
    getitem_7: "f32[8, 784, 1]" = var_mean_2[0]
    getitem_8: "f32[8, 784, 1]" = var_mean_2[1];  var_mean_2 = None
    add_24: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
    rsqrt_2: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_9, getitem_8);  clone_9 = getitem_8 = None
    mul_34: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_2);  sub_7 = rsqrt_2 = None
    mul_35: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg126_1);  mul_34 = arg126_1 = None
    add_25: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_35, arg127_1);  mul_35 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[6272, 768]" = torch.ops.aten.view.default(add_25, [6272, 768]);  add_25 = None
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_1: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg129_1, view_18, permute_10);  arg129_1 = view_18 = permute_10 = None
    view_19: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_1, [8, 784, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_36: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_37: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf_3: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_26: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_38: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_26);  mul_36 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_10: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_20: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_10, [6272, 3072]);  clone_10 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_2: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg131_1, view_20, permute_11);  arg131_1 = view_20 = permute_11 = None
    view_21: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_2, [8, 784, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_11: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_21);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_39: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg3_1, clone_11);  arg3_1 = clone_11 = None
    add_27: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_23, mul_39);  add_23 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_12: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 784, 1]" = var_mean_3[0]
    getitem_10: "f32[8, 784, 1]" = var_mean_3[1];  var_mean_3 = None
    add_28: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_3: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_10);  clone_12 = getitem_10 = None
    mul_40: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
    mul_41: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg132_1);  mul_40 = arg132_1 = None
    add_29: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_41, arg133_1);  mul_41 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_22: "f32[6272, 768]" = torch.ops.aten.view.default(add_29, [6272, 768]);  add_29 = None
    permute_12: "f32[768, 2304]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_3: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg135_1, view_22, permute_12);  arg135_1 = view_22 = permute_12 = None
    view_23: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_3, [8, 784, 2304]);  addmm_3 = None
    view_24: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_23, [8, 784, 3, 16, 48]);  view_23 = None
    permute_13: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_24, [2, 0, 3, 4, 1]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_13);  permute_13 = None
    getitem_11: "f32[8, 16, 48, 784]" = unbind_1[0]
    getitem_12: "f32[8, 16, 48, 784]" = unbind_1[1]
    getitem_13: "f32[8, 16, 48, 784]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_6: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_11, 2.0)
    sum_4: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_6, [-1], True);  pow_6 = None
    pow_7: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
    clamp_min_2: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_7, 1e-12);  pow_7 = None
    expand_6: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_2, [8, 16, 48, 784]);  clamp_min_2 = None
    div_9: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_11, expand_6);  getitem_11 = expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_8: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_12, 2.0)
    sum_5: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_8, [-1], True);  pow_8 = None
    pow_9: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_5, 0.5);  sum_5 = None
    clamp_min_3: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_9, 1e-12);  pow_9 = None
    expand_7: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_3, [8, 16, 48, 784]);  clamp_min_3 = None
    div_10: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_12, expand_7);  getitem_12 = expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_14: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_10, [0, 1, 3, 2]);  div_10 = None
    expand_8: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_9, [8, 16, 48, 784]);  div_9 = None
    clone_13: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_25: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_13, [128, 48, 784]);  clone_13 = None
    expand_9: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_14, [8, 16, 784, 48]);  permute_14 = None
    clone_14: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_26: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_14, [128, 784, 48]);  clone_14 = None
    bmm_2: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_25, view_26);  view_25 = view_26 = None
    view_27: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_2, [8, 16, 48, 48]);  bmm_2 = None
    mul_42: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_27, arg5_1);  view_27 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_42, [-1], True)
    sub_9: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_42, amax_1);  mul_42 = amax_1 = None
    exp_1: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_6: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_11: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_1, sum_6);  exp_1 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_15: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_10: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_15, [8, 16, 48, 48]);  clone_15 = None
    view_28: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_10, [128, 48, 48]);  expand_10 = None
    expand_11: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_13, [8, 16, 48, 784]);  getitem_13 = None
    clone_16: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_29: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_16, [128, 48, 784]);  clone_16 = None
    bmm_3: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_28, view_29);  view_28 = view_29 = None
    view_30: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_3, [8, 16, 48, 784]);  bmm_3 = None
    permute_15: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_30, [0, 3, 1, 2]);  view_30 = None
    view_31: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_15, [8, 784, 768]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    clone_17: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_31, memory_format = torch.contiguous_format);  view_31 = None
    view_32: "f32[6272, 768]" = torch.ops.aten.view.default(clone_17, [6272, 768]);  clone_17 = None
    mm_1: "f32[6272, 768]" = torch.ops.aten.mm.default(view_32, permute_16);  view_32 = permute_16 = None
    view_33: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_1, [8, 784, 768]);  mm_1 = None
    add_30: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_33, arg137_1);  view_33 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_18: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_43: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg4_1, clone_18);  arg4_1 = clone_18 = None
    add_31: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_27, mul_43);  add_27 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_19: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_19, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 784, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 784, 1]" = var_mean_4[1];  var_mean_4 = None
    add_32: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_10: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_19, getitem_15);  clone_19 = getitem_15 = None
    mul_44: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_4);  sub_10 = rsqrt_4 = None
    mul_45: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg138_1);  mul_44 = arg138_1 = None
    add_33: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_45, arg139_1);  mul_45 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_17: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_33, [0, 2, 1]);  add_33 = None
    view_34: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_17, [8, 768, 28, 28]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_6: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_34, arg140_1, arg141_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_34 = arg140_1 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_46: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.5)
    mul_47: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476);  convolution_6 = None
    erf_4: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_34: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_48: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, add_34);  mul_46 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_14: "f32[768]" = torch.ops.prims.convert_element_type.default(arg640_1, torch.float32);  arg640_1 = None
    convert_element_type_15: "f32[768]" = torch.ops.prims.convert_element_type.default(arg641_1, torch.float32);  arg641_1 = None
    add_35: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_4: "f32[768]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_4: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_49: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_39: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_40: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, -1);  unsqueeze_39 = None
    unsqueeze_41: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_49, -1);  mul_49 = None
    unsqueeze_42: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, -1);  unsqueeze_41 = None
    sub_11: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_48, unsqueeze_40);  mul_48 = unsqueeze_40 = None
    mul_50: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_42);  sub_11 = unsqueeze_42 = None
    unsqueeze_43: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_44: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, -1);  unsqueeze_43 = None
    mul_51: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_44);  mul_50 = unsqueeze_44 = None
    unsqueeze_45: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_46: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, -1);  unsqueeze_45 = None
    add_36: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_51, unsqueeze_46);  mul_51 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_7: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_36, arg144_1, arg145_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_36 = arg144_1 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_35: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_7, [8, 768, 784]);  convolution_7 = None
    permute_18: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_52: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg6_1, permute_18);  arg6_1 = permute_18 = None
    add_37: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_31, mul_52);  add_31 = mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_20: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_37, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_20, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 784, 1]" = var_mean_5[0]
    getitem_17: "f32[8, 784, 1]" = var_mean_5[1];  var_mean_5 = None
    add_38: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_5: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_12: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_20, getitem_17);  clone_20 = getitem_17 = None
    mul_53: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = rsqrt_5 = None
    mul_54: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_53, arg146_1);  mul_53 = arg146_1 = None
    add_39: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_54, arg147_1);  mul_54 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_36: "f32[6272, 768]" = torch.ops.aten.view.default(add_39, [6272, 768]);  add_39 = None
    permute_19: "f32[768, 3072]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_4: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg149_1, view_36, permute_19);  arg149_1 = view_36 = permute_19 = None
    view_37: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_4, [8, 784, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_55: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_37, 0.5)
    mul_56: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_37, 0.7071067811865476);  view_37 = None
    erf_5: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_40: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_57: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_55, add_40);  mul_55 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_21: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_38: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_21, [6272, 3072]);  clone_21 = None
    permute_20: "f32[3072, 768]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_5: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg151_1, view_38, permute_20);  arg151_1 = view_38 = permute_20 = None
    view_39: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_5, [8, 784, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_22: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_39);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_58: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg7_1, clone_22);  arg7_1 = clone_22 = None
    add_41: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_37, mul_58);  add_37 = mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_23: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_23, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 784, 1]" = var_mean_6[0]
    getitem_19: "f32[8, 784, 1]" = var_mean_6[1];  var_mean_6 = None
    add_42: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_6: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_13: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_23, getitem_19);  clone_23 = getitem_19 = None
    mul_59: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_6);  sub_13 = rsqrt_6 = None
    mul_60: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg152_1);  mul_59 = arg152_1 = None
    add_43: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_60, arg153_1);  mul_60 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_40: "f32[6272, 768]" = torch.ops.aten.view.default(add_43, [6272, 768]);  add_43 = None
    permute_21: "f32[768, 2304]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_6: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg155_1, view_40, permute_21);  arg155_1 = view_40 = permute_21 = None
    view_41: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_6, [8, 784, 2304]);  addmm_6 = None
    view_42: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_41, [8, 784, 3, 16, 48]);  view_41 = None
    permute_22: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_42, [2, 0, 3, 4, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_22);  permute_22 = None
    getitem_20: "f32[8, 16, 48, 784]" = unbind_2[0]
    getitem_21: "f32[8, 16, 48, 784]" = unbind_2[1]
    getitem_22: "f32[8, 16, 48, 784]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_10: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_20, 2.0)
    sum_7: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_10, [-1], True);  pow_10 = None
    pow_11: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_7, 0.5);  sum_7 = None
    clamp_min_4: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_11, 1e-12);  pow_11 = None
    expand_12: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_4, [8, 16, 48, 784]);  clamp_min_4 = None
    div_12: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_20, expand_12);  getitem_20 = expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_12: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_21, 2.0)
    sum_8: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_12, [-1], True);  pow_12 = None
    pow_13: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None
    clamp_min_5: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_13, 1e-12);  pow_13 = None
    expand_13: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_5, [8, 16, 48, 784]);  clamp_min_5 = None
    div_13: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_21, expand_13);  getitem_21 = expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_23: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_13, [0, 1, 3, 2]);  div_13 = None
    expand_14: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_12, [8, 16, 48, 784]);  div_12 = None
    clone_24: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_43: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_24, [128, 48, 784]);  clone_24 = None
    expand_15: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_23, [8, 16, 784, 48]);  permute_23 = None
    clone_25: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_44: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_25, [128, 784, 48]);  clone_25 = None
    bmm_4: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_4, [8, 16, 48, 48]);  bmm_4 = None
    mul_61: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_45, arg9_1);  view_45 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_61, [-1], True)
    sub_14: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_61, amax_2);  mul_61 = amax_2 = None
    exp_2: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_9: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_14: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_2, sum_9);  exp_2 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_26: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_16: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_26, [8, 16, 48, 48]);  clone_26 = None
    view_46: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_16, [128, 48, 48]);  expand_16 = None
    expand_17: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_22, [8, 16, 48, 784]);  getitem_22 = None
    clone_27: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_47: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_27, [128, 48, 784]);  clone_27 = None
    bmm_5: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_46, view_47);  view_46 = view_47 = None
    view_48: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_5, [8, 16, 48, 784]);  bmm_5 = None
    permute_24: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_48, [0, 3, 1, 2]);  view_48 = None
    view_49: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_24, [8, 784, 768]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    clone_28: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_49, memory_format = torch.contiguous_format);  view_49 = None
    view_50: "f32[6272, 768]" = torch.ops.aten.view.default(clone_28, [6272, 768]);  clone_28 = None
    mm_2: "f32[6272, 768]" = torch.ops.aten.mm.default(view_50, permute_25);  view_50 = permute_25 = None
    view_51: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_2, [8, 784, 768]);  mm_2 = None
    add_44: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_51, arg157_1);  view_51 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_29: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_62: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg8_1, clone_29);  arg8_1 = clone_29 = None
    add_45: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_41, mul_62);  add_41 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_30: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_45, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
    getitem_23: "f32[8, 784, 1]" = var_mean_7[0]
    getitem_24: "f32[8, 784, 1]" = var_mean_7[1];  var_mean_7 = None
    add_46: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-06);  getitem_23 = None
    rsqrt_7: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_15: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_30, getitem_24);  clone_30 = getitem_24 = None
    mul_63: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_7);  sub_15 = rsqrt_7 = None
    mul_64: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg158_1);  mul_63 = arg158_1 = None
    add_47: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_64, arg159_1);  mul_64 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_26: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_47, [0, 2, 1]);  add_47 = None
    view_52: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_26, [8, 768, 28, 28]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_8: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_52, arg160_1, arg161_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_52 = arg160_1 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_65: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.5)
    mul_66: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.7071067811865476);  convolution_8 = None
    erf_6: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_66);  mul_66 = None
    add_48: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_67: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, add_48);  mul_65 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_16: "f32[768]" = torch.ops.prims.convert_element_type.default(arg643_1, torch.float32);  arg643_1 = None
    convert_element_type_17: "f32[768]" = torch.ops.prims.convert_element_type.default(arg644_1, torch.float32);  arg644_1 = None
    add_49: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_5: "f32[768]" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
    reciprocal_5: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_68: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_47: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_48: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_47, -1);  unsqueeze_47 = None
    unsqueeze_49: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_68, -1);  mul_68 = None
    unsqueeze_50: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, -1);  unsqueeze_49 = None
    sub_16: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_67, unsqueeze_48);  mul_67 = unsqueeze_48 = None
    mul_69: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_50);  sub_16 = unsqueeze_50 = None
    unsqueeze_51: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
    unsqueeze_52: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, -1);  unsqueeze_51 = None
    mul_70: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_69, unsqueeze_52);  mul_69 = unsqueeze_52 = None
    unsqueeze_53: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
    unsqueeze_54: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, -1);  unsqueeze_53 = None
    add_50: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_54);  mul_70 = unsqueeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_9: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_50, arg164_1, arg165_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_50 = arg164_1 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_53: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_9, [8, 768, 784]);  convolution_9 = None
    permute_27: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_71: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg10_1, permute_27);  arg10_1 = permute_27 = None
    add_51: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_45, mul_71);  add_45 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_31: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_51, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_31, [2], correction = 0, keepdim = True)
    getitem_25: "f32[8, 784, 1]" = var_mean_8[0]
    getitem_26: "f32[8, 784, 1]" = var_mean_8[1];  var_mean_8 = None
    add_52: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-06);  getitem_25 = None
    rsqrt_8: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_17: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_31, getitem_26);  clone_31 = getitem_26 = None
    mul_72: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_8);  sub_17 = rsqrt_8 = None
    mul_73: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg166_1);  mul_72 = arg166_1 = None
    add_53: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_73, arg167_1);  mul_73 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_54: "f32[6272, 768]" = torch.ops.aten.view.default(add_53, [6272, 768]);  add_53 = None
    permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_7: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg169_1, view_54, permute_28);  arg169_1 = view_54 = permute_28 = None
    view_55: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_7, [8, 784, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_55, 0.5)
    mul_75: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_55, 0.7071067811865476);  view_55 = None
    erf_7: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_54: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_76: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_54);  mul_74 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_56: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_32, [6272, 3072]);  clone_32 = None
    permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_8: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg171_1, view_56, permute_29);  arg171_1 = view_56 = permute_29 = None
    view_57: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_8, [8, 784, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_57);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_77: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg11_1, clone_33);  arg11_1 = clone_33 = None
    add_55: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_51, mul_77);  add_51 = mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_34: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_34, [2], correction = 0, keepdim = True)
    getitem_27: "f32[8, 784, 1]" = var_mean_9[0]
    getitem_28: "f32[8, 784, 1]" = var_mean_9[1];  var_mean_9 = None
    add_56: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_27, 1e-06);  getitem_27 = None
    rsqrt_9: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_18: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_34, getitem_28);  clone_34 = getitem_28 = None
    mul_78: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_9);  sub_18 = rsqrt_9 = None
    mul_79: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg172_1);  mul_78 = arg172_1 = None
    add_57: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_79, arg173_1);  mul_79 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_58: "f32[6272, 768]" = torch.ops.aten.view.default(add_57, [6272, 768]);  add_57 = None
    permute_30: "f32[768, 2304]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_9: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg175_1, view_58, permute_30);  arg175_1 = view_58 = permute_30 = None
    view_59: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_9, [8, 784, 2304]);  addmm_9 = None
    view_60: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_59, [8, 784, 3, 16, 48]);  view_59 = None
    permute_31: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_60, [2, 0, 3, 4, 1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
    getitem_29: "f32[8, 16, 48, 784]" = unbind_3[0]
    getitem_30: "f32[8, 16, 48, 784]" = unbind_3[1]
    getitem_31: "f32[8, 16, 48, 784]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_14: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_29, 2.0)
    sum_10: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_14, [-1], True);  pow_14 = None
    pow_15: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None
    clamp_min_6: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_15, 1e-12);  pow_15 = None
    expand_18: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_6, [8, 16, 48, 784]);  clamp_min_6 = None
    div_15: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_29, expand_18);  getitem_29 = expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_16: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_30, 2.0)
    sum_11: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_16, [-1], True);  pow_16 = None
    pow_17: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_11, 0.5);  sum_11 = None
    clamp_min_7: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_17, 1e-12);  pow_17 = None
    expand_19: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_7, [8, 16, 48, 784]);  clamp_min_7 = None
    div_16: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_30, expand_19);  getitem_30 = expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_32: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_16, [0, 1, 3, 2]);  div_16 = None
    expand_20: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_15, [8, 16, 48, 784]);  div_15 = None
    clone_35: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_61: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_35, [128, 48, 784]);  clone_35 = None
    expand_21: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_32, [8, 16, 784, 48]);  permute_32 = None
    clone_36: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_62: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_36, [128, 784, 48]);  clone_36 = None
    bmm_6: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_61, view_62);  view_61 = view_62 = None
    view_63: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_6, [8, 16, 48, 48]);  bmm_6 = None
    mul_80: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_63, arg13_1);  view_63 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_80, [-1], True)
    sub_19: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_80, amax_3);  mul_80 = amax_3 = None
    exp_3: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_12: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_17: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_3, sum_12);  exp_3 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_37: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_22: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_37, [8, 16, 48, 48]);  clone_37 = None
    view_64: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_22, [128, 48, 48]);  expand_22 = None
    expand_23: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_31, [8, 16, 48, 784]);  getitem_31 = None
    clone_38: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_65: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_38, [128, 48, 784]);  clone_38 = None
    bmm_7: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
    view_66: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_7, [8, 16, 48, 784]);  bmm_7 = None
    permute_33: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_66, [0, 3, 1, 2]);  view_66 = None
    view_67: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_33, [8, 784, 768]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    clone_39: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_67, memory_format = torch.contiguous_format);  view_67 = None
    view_68: "f32[6272, 768]" = torch.ops.aten.view.default(clone_39, [6272, 768]);  clone_39 = None
    mm_3: "f32[6272, 768]" = torch.ops.aten.mm.default(view_68, permute_34);  view_68 = permute_34 = None
    view_69: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_3, [8, 784, 768]);  mm_3 = None
    add_58: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_69, arg177_1);  view_69 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_40: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_58);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_81: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg12_1, clone_40);  arg12_1 = clone_40 = None
    add_59: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_55, mul_81);  add_55 = mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_41: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_41, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 784, 1]" = var_mean_10[0]
    getitem_33: "f32[8, 784, 1]" = var_mean_10[1];  var_mean_10 = None
    add_60: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_10: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_20: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_41, getitem_33);  clone_41 = getitem_33 = None
    mul_82: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_10);  sub_20 = rsqrt_10 = None
    mul_83: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg178_1);  mul_82 = arg178_1 = None
    add_61: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_83, arg179_1);  mul_83 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_35: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_61, [0, 2, 1]);  add_61 = None
    view_70: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_35, [8, 768, 28, 28]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_10: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_70, arg180_1, arg181_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_70 = arg180_1 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_84: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, 0.5)
    mul_85: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, 0.7071067811865476);  convolution_10 = None
    erf_8: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_62: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_86: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_84, add_62);  mul_84 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_18: "f32[768]" = torch.ops.prims.convert_element_type.default(arg646_1, torch.float32);  arg646_1 = None
    convert_element_type_19: "f32[768]" = torch.ops.prims.convert_element_type.default(arg647_1, torch.float32);  arg647_1 = None
    add_63: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_6: "f32[768]" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
    reciprocal_6: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_87: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_55: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_56: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, -1);  unsqueeze_55 = None
    unsqueeze_57: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_58: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, -1);  unsqueeze_57 = None
    sub_21: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_86, unsqueeze_56);  mul_86 = unsqueeze_56 = None
    mul_88: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_58);  sub_21 = unsqueeze_58 = None
    unsqueeze_59: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
    unsqueeze_60: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, -1);  unsqueeze_59 = None
    mul_89: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_60);  mul_88 = unsqueeze_60 = None
    unsqueeze_61: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
    unsqueeze_62: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, -1);  unsqueeze_61 = None
    add_64: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_62);  mul_89 = unsqueeze_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_11: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_64, arg184_1, arg185_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_64 = arg184_1 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_71: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_11, [8, 768, 784]);  convolution_11 = None
    permute_36: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_90: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg14_1, permute_36);  arg14_1 = permute_36 = None
    add_65: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_59, mul_90);  add_59 = mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_42: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_65, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 784, 1]" = var_mean_11[0]
    getitem_35: "f32[8, 784, 1]" = var_mean_11[1];  var_mean_11 = None
    add_66: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_11: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_22: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_35);  clone_42 = getitem_35 = None
    mul_91: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_11);  sub_22 = rsqrt_11 = None
    mul_92: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_91, arg186_1);  mul_91 = arg186_1 = None
    add_67: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_92, arg187_1);  mul_92 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_72: "f32[6272, 768]" = torch.ops.aten.view.default(add_67, [6272, 768]);  add_67 = None
    permute_37: "f32[768, 3072]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_10: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg189_1, view_72, permute_37);  arg189_1 = view_72 = permute_37 = None
    view_73: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 784, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.5)
    mul_94: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
    erf_9: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_68: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_95: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_68);  mul_93 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_43: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_74: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_43, [6272, 3072]);  clone_43 = None
    permute_38: "f32[3072, 768]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_11: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg191_1, view_74, permute_38);  arg191_1 = view_74 = permute_38 = None
    view_75: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_11, [8, 784, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_44: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_75);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_96: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg15_1, clone_44);  arg15_1 = clone_44 = None
    add_69: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_65, mul_96);  add_65 = mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_45: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 784, 1]" = var_mean_12[0]
    getitem_37: "f32[8, 784, 1]" = var_mean_12[1];  var_mean_12 = None
    add_70: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_12: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_23: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_45, getitem_37);  clone_45 = getitem_37 = None
    mul_97: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_12);  sub_23 = rsqrt_12 = None
    mul_98: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_97, arg192_1);  mul_97 = arg192_1 = None
    add_71: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_98, arg193_1);  mul_98 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_76: "f32[6272, 768]" = torch.ops.aten.view.default(add_71, [6272, 768]);  add_71 = None
    permute_39: "f32[768, 2304]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_12: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg195_1, view_76, permute_39);  arg195_1 = view_76 = permute_39 = None
    view_77: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 784, 2304]);  addmm_12 = None
    view_78: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_77, [8, 784, 3, 16, 48]);  view_77 = None
    permute_40: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_78, [2, 0, 3, 4, 1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_40);  permute_40 = None
    getitem_38: "f32[8, 16, 48, 784]" = unbind_4[0]
    getitem_39: "f32[8, 16, 48, 784]" = unbind_4[1]
    getitem_40: "f32[8, 16, 48, 784]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_18: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_38, 2.0)
    sum_13: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_18, [-1], True);  pow_18 = None
    pow_19: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_13, 0.5);  sum_13 = None
    clamp_min_8: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_19, 1e-12);  pow_19 = None
    expand_24: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_8, [8, 16, 48, 784]);  clamp_min_8 = None
    div_18: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_38, expand_24);  getitem_38 = expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_20: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_39, 2.0)
    sum_14: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_20, [-1], True);  pow_20 = None
    pow_21: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None
    clamp_min_9: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_21, 1e-12);  pow_21 = None
    expand_25: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_9, [8, 16, 48, 784]);  clamp_min_9 = None
    div_19: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_39, expand_25);  getitem_39 = expand_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_41: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_19, [0, 1, 3, 2]);  div_19 = None
    expand_26: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_18, [8, 16, 48, 784]);  div_18 = None
    clone_46: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_79: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_46, [128, 48, 784]);  clone_46 = None
    expand_27: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_41, [8, 16, 784, 48]);  permute_41 = None
    clone_47: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_80: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_47, [128, 784, 48]);  clone_47 = None
    bmm_8: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_79, view_80);  view_79 = view_80 = None
    view_81: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_8, [8, 16, 48, 48]);  bmm_8 = None
    mul_99: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_81, arg17_1);  view_81 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_99, [-1], True)
    sub_24: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_99, amax_4);  mul_99 = amax_4 = None
    exp_4: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_15: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_20: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_4, sum_15);  exp_4 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_48: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_28: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_48, [8, 16, 48, 48]);  clone_48 = None
    view_82: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_28, [128, 48, 48]);  expand_28 = None
    expand_29: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_40, [8, 16, 48, 784]);  getitem_40 = None
    clone_49: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_83: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_49, [128, 48, 784]);  clone_49 = None
    bmm_9: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
    view_84: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_9, [8, 16, 48, 784]);  bmm_9 = None
    permute_42: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_84, [0, 3, 1, 2]);  view_84 = None
    view_85: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_42, [8, 784, 768]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_43: "f32[768, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    clone_50: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_85, memory_format = torch.contiguous_format);  view_85 = None
    view_86: "f32[6272, 768]" = torch.ops.aten.view.default(clone_50, [6272, 768]);  clone_50 = None
    mm_4: "f32[6272, 768]" = torch.ops.aten.mm.default(view_86, permute_43);  view_86 = permute_43 = None
    view_87: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_4, [8, 784, 768]);  mm_4 = None
    add_72: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_87, arg197_1);  view_87 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_51: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_100: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg16_1, clone_51);  arg16_1 = clone_51 = None
    add_73: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_69, mul_100);  add_69 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_52: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_52, [2], correction = 0, keepdim = True)
    getitem_41: "f32[8, 784, 1]" = var_mean_13[0]
    getitem_42: "f32[8, 784, 1]" = var_mean_13[1];  var_mean_13 = None
    add_74: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-06);  getitem_41 = None
    rsqrt_13: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_25: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_52, getitem_42);  clone_52 = getitem_42 = None
    mul_101: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_13);  sub_25 = rsqrt_13 = None
    mul_102: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_101, arg198_1);  mul_101 = arg198_1 = None
    add_75: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_102, arg199_1);  mul_102 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_44: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_75, [0, 2, 1]);  add_75 = None
    view_88: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_44, [8, 768, 28, 28]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_12: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_88, arg200_1, arg201_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_88 = arg200_1 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_103: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
    mul_104: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476);  convolution_12 = None
    erf_10: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_76: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_105: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, add_76);  mul_103 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_20: "f32[768]" = torch.ops.prims.convert_element_type.default(arg649_1, torch.float32);  arg649_1 = None
    convert_element_type_21: "f32[768]" = torch.ops.prims.convert_element_type.default(arg650_1, torch.float32);  arg650_1 = None
    add_77: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_7: "f32[768]" = torch.ops.aten.sqrt.default(add_77);  add_77 = None
    reciprocal_7: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_106: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_63: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_64: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, -1);  unsqueeze_63 = None
    unsqueeze_65: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_106, -1);  mul_106 = None
    unsqueeze_66: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, -1);  unsqueeze_65 = None
    sub_26: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_105, unsqueeze_64);  mul_105 = unsqueeze_64 = None
    mul_107: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_66);  sub_26 = unsqueeze_66 = None
    unsqueeze_67: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
    unsqueeze_68: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, -1);  unsqueeze_67 = None
    mul_108: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_68);  mul_107 = unsqueeze_68 = None
    unsqueeze_69: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
    unsqueeze_70: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, -1);  unsqueeze_69 = None
    add_78: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_70);  mul_108 = unsqueeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_13: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_78, arg204_1, arg205_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_78 = arg204_1 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_89: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_13, [8, 768, 784]);  convolution_13 = None
    permute_45: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_109: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg18_1, permute_45);  arg18_1 = permute_45 = None
    add_79: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_73, mul_109);  add_73 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_53: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_79, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
    getitem_43: "f32[8, 784, 1]" = var_mean_14[0]
    getitem_44: "f32[8, 784, 1]" = var_mean_14[1];  var_mean_14 = None
    add_80: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-06);  getitem_43 = None
    rsqrt_14: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_27: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_53, getitem_44);  clone_53 = getitem_44 = None
    mul_110: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_14);  sub_27 = rsqrt_14 = None
    mul_111: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_110, arg206_1);  mul_110 = arg206_1 = None
    add_81: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_111, arg207_1);  mul_111 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_90: "f32[6272, 768]" = torch.ops.aten.view.default(add_81, [6272, 768]);  add_81 = None
    permute_46: "f32[768, 3072]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_13: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg209_1, view_90, permute_46);  arg209_1 = view_90 = permute_46 = None
    view_91: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_13, [8, 784, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_112: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_91, 0.5)
    mul_113: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476);  view_91 = None
    erf_11: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_82: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_114: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_112, add_82);  mul_112 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_114);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_92: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_54, [6272, 3072]);  clone_54 = None
    permute_47: "f32[3072, 768]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_14: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg211_1, view_92, permute_47);  arg211_1 = view_92 = permute_47 = None
    view_93: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_14, [8, 784, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_93);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_115: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg19_1, clone_55);  arg19_1 = clone_55 = None
    add_83: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_79, mul_115);  add_79 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_56: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
    getitem_45: "f32[8, 784, 1]" = var_mean_15[0]
    getitem_46: "f32[8, 784, 1]" = var_mean_15[1];  var_mean_15 = None
    add_84: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-06);  getitem_45 = None
    rsqrt_15: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_28: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_56, getitem_46);  clone_56 = getitem_46 = None
    mul_116: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_15);  sub_28 = rsqrt_15 = None
    mul_117: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg212_1);  mul_116 = arg212_1 = None
    add_85: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_117, arg213_1);  mul_117 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_94: "f32[6272, 768]" = torch.ops.aten.view.default(add_85, [6272, 768]);  add_85 = None
    permute_48: "f32[768, 2304]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_15: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg215_1, view_94, permute_48);  arg215_1 = view_94 = permute_48 = None
    view_95: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_15, [8, 784, 2304]);  addmm_15 = None
    view_96: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_95, [8, 784, 3, 16, 48]);  view_95 = None
    permute_49: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_96, [2, 0, 3, 4, 1]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_49);  permute_49 = None
    getitem_47: "f32[8, 16, 48, 784]" = unbind_5[0]
    getitem_48: "f32[8, 16, 48, 784]" = unbind_5[1]
    getitem_49: "f32[8, 16, 48, 784]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_22: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_47, 2.0)
    sum_16: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_22, [-1], True);  pow_22 = None
    pow_23: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_16, 0.5);  sum_16 = None
    clamp_min_10: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_23, 1e-12);  pow_23 = None
    expand_30: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_10, [8, 16, 48, 784]);  clamp_min_10 = None
    div_21: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_47, expand_30);  getitem_47 = expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_24: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_48, 2.0)
    sum_17: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_24, [-1], True);  pow_24 = None
    pow_25: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_17, 0.5);  sum_17 = None
    clamp_min_11: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_25, 1e-12);  pow_25 = None
    expand_31: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_11, [8, 16, 48, 784]);  clamp_min_11 = None
    div_22: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_48, expand_31);  getitem_48 = expand_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_50: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_22, [0, 1, 3, 2]);  div_22 = None
    expand_32: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_21, [8, 16, 48, 784]);  div_21 = None
    clone_57: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_97: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_57, [128, 48, 784]);  clone_57 = None
    expand_33: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_50, [8, 16, 784, 48]);  permute_50 = None
    clone_58: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_98: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_58, [128, 784, 48]);  clone_58 = None
    bmm_10: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_10, [8, 16, 48, 48]);  bmm_10 = None
    mul_118: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_99, arg21_1);  view_99 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_118, [-1], True)
    sub_29: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_118, amax_5);  mul_118 = amax_5 = None
    exp_5: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_18: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_23: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_5, sum_18);  exp_5 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_59: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_59, [8, 16, 48, 48]);  clone_59 = None
    view_100: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_34, [128, 48, 48]);  expand_34 = None
    expand_35: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_49, [8, 16, 48, 784]);  getitem_49 = None
    clone_60: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_101: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_60, [128, 48, 784]);  clone_60 = None
    bmm_11: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
    view_102: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_11, [8, 16, 48, 784]);  bmm_11 = None
    permute_51: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_102, [0, 3, 1, 2]);  view_102 = None
    view_103: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_51, [8, 784, 768]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    clone_61: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_103, memory_format = torch.contiguous_format);  view_103 = None
    view_104: "f32[6272, 768]" = torch.ops.aten.view.default(clone_61, [6272, 768]);  clone_61 = None
    mm_5: "f32[6272, 768]" = torch.ops.aten.mm.default(view_104, permute_52);  view_104 = permute_52 = None
    view_105: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_5, [8, 784, 768]);  mm_5 = None
    add_86: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_105, arg217_1);  view_105 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_62: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_119: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg20_1, clone_62);  arg20_1 = clone_62 = None
    add_87: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_83, mul_119);  add_83 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_63: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_50: "f32[8, 784, 1]" = var_mean_16[0]
    getitem_51: "f32[8, 784, 1]" = var_mean_16[1];  var_mean_16 = None
    add_88: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_16: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_30: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_63, getitem_51);  clone_63 = getitem_51 = None
    mul_120: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_16);  sub_30 = rsqrt_16 = None
    mul_121: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_120, arg218_1);  mul_120 = arg218_1 = None
    add_89: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_121, arg219_1);  mul_121 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_53: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_89, [0, 2, 1]);  add_89 = None
    view_106: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_53, [8, 768, 28, 28]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_14: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_106, arg220_1, arg221_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_106 = arg220_1 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_122: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_123: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476);  convolution_14 = None
    erf_12: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_90: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_124: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_122, add_90);  mul_122 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_22: "f32[768]" = torch.ops.prims.convert_element_type.default(arg652_1, torch.float32);  arg652_1 = None
    convert_element_type_23: "f32[768]" = torch.ops.prims.convert_element_type.default(arg653_1, torch.float32);  arg653_1 = None
    add_91: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_8: "f32[768]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    reciprocal_8: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_125: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_71: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_72: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, -1);  unsqueeze_71 = None
    unsqueeze_73: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_125, -1);  mul_125 = None
    unsqueeze_74: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, -1);  unsqueeze_73 = None
    sub_31: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_124, unsqueeze_72);  mul_124 = unsqueeze_72 = None
    mul_126: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_74);  sub_31 = unsqueeze_74 = None
    unsqueeze_75: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
    unsqueeze_76: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, -1);  unsqueeze_75 = None
    mul_127: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_76);  mul_126 = unsqueeze_76 = None
    unsqueeze_77: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
    unsqueeze_78: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, -1);  unsqueeze_77 = None
    add_92: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_78);  mul_127 = unsqueeze_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_15: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_92, arg224_1, arg225_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_92 = arg224_1 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_107: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_15, [8, 768, 784]);  convolution_15 = None
    permute_54: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_128: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg22_1, permute_54);  arg22_1 = permute_54 = None
    add_93: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_87, mul_128);  add_87 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_64: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_93, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_64, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 784, 1]" = var_mean_17[0]
    getitem_53: "f32[8, 784, 1]" = var_mean_17[1];  var_mean_17 = None
    add_94: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_17: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_32: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_64, getitem_53);  clone_64 = getitem_53 = None
    mul_129: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_17);  sub_32 = rsqrt_17 = None
    mul_130: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_129, arg226_1);  mul_129 = arg226_1 = None
    add_95: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_130, arg227_1);  mul_130 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_108: "f32[6272, 768]" = torch.ops.aten.view.default(add_95, [6272, 768]);  add_95 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    addmm_16: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg229_1, view_108, permute_55);  arg229_1 = view_108 = permute_55 = None
    view_109: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_16, [8, 784, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_131: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_132: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_13: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_96: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_133: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_131, add_96);  mul_131 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_65: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_133);  mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_110: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_65, [6272, 3072]);  clone_65 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_17: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg231_1, view_110, permute_56);  arg231_1 = view_110 = permute_56 = None
    view_111: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_17, [8, 784, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_66: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_111);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_134: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg23_1, clone_66);  arg23_1 = clone_66 = None
    add_97: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_93, mul_134);  add_93 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_67: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 784, 1]" = var_mean_18[0]
    getitem_55: "f32[8, 784, 1]" = var_mean_18[1];  var_mean_18 = None
    add_98: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_18: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_33: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_67, getitem_55);  clone_67 = getitem_55 = None
    mul_135: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_18);  sub_33 = rsqrt_18 = None
    mul_136: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_135, arg232_1);  mul_135 = arg232_1 = None
    add_99: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_136, arg233_1);  mul_136 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_112: "f32[6272, 768]" = torch.ops.aten.view.default(add_99, [6272, 768]);  add_99 = None
    permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_18: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg235_1, view_112, permute_57);  arg235_1 = view_112 = permute_57 = None
    view_113: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_18, [8, 784, 2304]);  addmm_18 = None
    view_114: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_113, [8, 784, 3, 16, 48]);  view_113 = None
    permute_58: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_114, [2, 0, 3, 4, 1]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_56: "f32[8, 16, 48, 784]" = unbind_6[0]
    getitem_57: "f32[8, 16, 48, 784]" = unbind_6[1]
    getitem_58: "f32[8, 16, 48, 784]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_26: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_56, 2.0)
    sum_19: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_26, [-1], True);  pow_26 = None
    pow_27: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_19, 0.5);  sum_19 = None
    clamp_min_12: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_27, 1e-12);  pow_27 = None
    expand_36: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_12, [8, 16, 48, 784]);  clamp_min_12 = None
    div_24: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_56, expand_36);  getitem_56 = expand_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_28: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_57, 2.0)
    sum_20: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_28, [-1], True);  pow_28 = None
    pow_29: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_20, 0.5);  sum_20 = None
    clamp_min_13: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_29, 1e-12);  pow_29 = None
    expand_37: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_13, [8, 16, 48, 784]);  clamp_min_13 = None
    div_25: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_57, expand_37);  getitem_57 = expand_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_59: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_25, [0, 1, 3, 2]);  div_25 = None
    expand_38: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_24, [8, 16, 48, 784]);  div_24 = None
    clone_68: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_115: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_68, [128, 48, 784]);  clone_68 = None
    expand_39: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_59, [8, 16, 784, 48]);  permute_59 = None
    clone_69: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_116: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_69, [128, 784, 48]);  clone_69 = None
    bmm_12: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_115, view_116);  view_115 = view_116 = None
    view_117: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_12, [8, 16, 48, 48]);  bmm_12 = None
    mul_137: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_117, arg25_1);  view_117 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_137, [-1], True)
    sub_34: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_137, amax_6);  mul_137 = amax_6 = None
    exp_6: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_21: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_26: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_6, sum_21);  exp_6 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_70: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_40: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_70, [8, 16, 48, 48]);  clone_70 = None
    view_118: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_40, [128, 48, 48]);  expand_40 = None
    expand_41: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_58, [8, 16, 48, 784]);  getitem_58 = None
    clone_71: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_119: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_71, [128, 48, 784]);  clone_71 = None
    bmm_13: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_118, view_119);  view_118 = view_119 = None
    view_120: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_13, [8, 16, 48, 784]);  bmm_13 = None
    permute_60: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_120, [0, 3, 1, 2]);  view_120 = None
    view_121: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_60, [8, 784, 768]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    clone_72: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_121, memory_format = torch.contiguous_format);  view_121 = None
    view_122: "f32[6272, 768]" = torch.ops.aten.view.default(clone_72, [6272, 768]);  clone_72 = None
    mm_6: "f32[6272, 768]" = torch.ops.aten.mm.default(view_122, permute_61);  view_122 = permute_61 = None
    view_123: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_6, [8, 784, 768]);  mm_6 = None
    add_100: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_123, arg237_1);  view_123 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_73: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_138: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg24_1, clone_73);  arg24_1 = clone_73 = None
    add_101: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_97, mul_138);  add_97 = mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_74: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_74, [2], correction = 0, keepdim = True)
    getitem_59: "f32[8, 784, 1]" = var_mean_19[0]
    getitem_60: "f32[8, 784, 1]" = var_mean_19[1];  var_mean_19 = None
    add_102: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_59, 1e-06);  getitem_59 = None
    rsqrt_19: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_35: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_74, getitem_60);  clone_74 = getitem_60 = None
    mul_139: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_19);  sub_35 = rsqrt_19 = None
    mul_140: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_139, arg238_1);  mul_139 = arg238_1 = None
    add_103: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_140, arg239_1);  mul_140 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_62: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_103, [0, 2, 1]);  add_103 = None
    view_124: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_62, [8, 768, 28, 28]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_16: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_124, arg240_1, arg241_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_124 = arg240_1 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_141: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
    mul_142: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476);  convolution_16 = None
    erf_14: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
    add_104: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_143: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_141, add_104);  mul_141 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_24: "f32[768]" = torch.ops.prims.convert_element_type.default(arg655_1, torch.float32);  arg655_1 = None
    convert_element_type_25: "f32[768]" = torch.ops.prims.convert_element_type.default(arg656_1, torch.float32);  arg656_1 = None
    add_105: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_9: "f32[768]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_9: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_144: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_79: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_80: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, -1);  unsqueeze_79 = None
    unsqueeze_81: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_82: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, -1);  unsqueeze_81 = None
    sub_36: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_143, unsqueeze_80);  mul_143 = unsqueeze_80 = None
    mul_145: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_82);  sub_36 = unsqueeze_82 = None
    unsqueeze_83: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
    unsqueeze_84: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, -1);  unsqueeze_83 = None
    mul_146: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_84);  mul_145 = unsqueeze_84 = None
    unsqueeze_85: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
    unsqueeze_86: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, -1);  unsqueeze_85 = None
    add_106: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_86);  mul_146 = unsqueeze_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_17: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_106, arg244_1, arg245_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_106 = arg244_1 = arg245_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_125: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_17, [8, 768, 784]);  convolution_17 = None
    permute_63: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_147: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg26_1, permute_63);  arg26_1 = permute_63 = None
    add_107: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_101, mul_147);  add_101 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_75: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_107, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_75, [2], correction = 0, keepdim = True)
    getitem_61: "f32[8, 784, 1]" = var_mean_20[0]
    getitem_62: "f32[8, 784, 1]" = var_mean_20[1];  var_mean_20 = None
    add_108: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-06);  getitem_61 = None
    rsqrt_20: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_37: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_75, getitem_62);  clone_75 = getitem_62 = None
    mul_148: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_20);  sub_37 = rsqrt_20 = None
    mul_149: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_148, arg246_1);  mul_148 = arg246_1 = None
    add_109: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_149, arg247_1);  mul_149 = arg247_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[6272, 768]" = torch.ops.aten.view.default(add_109, [6272, 768]);  add_109 = None
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    addmm_19: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg249_1, view_126, permute_64);  arg249_1 = view_126 = permute_64 = None
    view_127: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_19, [8, 784, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_150: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_127, 0.5)
    mul_151: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476);  view_127 = None
    erf_15: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
    add_110: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_152: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_150, add_110);  mul_150 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_76: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_152);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_76, [6272, 3072]);  clone_76 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_20: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg251_1, view_128, permute_65);  arg251_1 = view_128 = permute_65 = None
    view_129: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_20, [8, 784, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_77: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_153: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg27_1, clone_77);  arg27_1 = clone_77 = None
    add_111: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_107, mul_153);  add_107 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_78: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_78, [2], correction = 0, keepdim = True)
    getitem_63: "f32[8, 784, 1]" = var_mean_21[0]
    getitem_64: "f32[8, 784, 1]" = var_mean_21[1];  var_mean_21 = None
    add_112: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-06);  getitem_63 = None
    rsqrt_21: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_38: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_78, getitem_64);  clone_78 = getitem_64 = None
    mul_154: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_21);  sub_38 = rsqrt_21 = None
    mul_155: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_154, arg252_1);  mul_154 = arg252_1 = None
    add_113: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_155, arg253_1);  mul_155 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_130: "f32[6272, 768]" = torch.ops.aten.view.default(add_113, [6272, 768]);  add_113 = None
    permute_66: "f32[768, 2304]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_21: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg255_1, view_130, permute_66);  arg255_1 = view_130 = permute_66 = None
    view_131: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_21, [8, 784, 2304]);  addmm_21 = None
    view_132: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_131, [8, 784, 3, 16, 48]);  view_131 = None
    permute_67: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_132, [2, 0, 3, 4, 1]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_65: "f32[8, 16, 48, 784]" = unbind_7[0]
    getitem_66: "f32[8, 16, 48, 784]" = unbind_7[1]
    getitem_67: "f32[8, 16, 48, 784]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_30: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_65, 2.0)
    sum_22: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_30, [-1], True);  pow_30 = None
    pow_31: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_22, 0.5);  sum_22 = None
    clamp_min_14: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_31, 1e-12);  pow_31 = None
    expand_42: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_14, [8, 16, 48, 784]);  clamp_min_14 = None
    div_27: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_65, expand_42);  getitem_65 = expand_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_32: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_66, 2.0)
    sum_23: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_32, [-1], True);  pow_32 = None
    pow_33: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_23, 0.5);  sum_23 = None
    clamp_min_15: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_33, 1e-12);  pow_33 = None
    expand_43: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_15, [8, 16, 48, 784]);  clamp_min_15 = None
    div_28: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_66, expand_43);  getitem_66 = expand_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_68: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_28, [0, 1, 3, 2]);  div_28 = None
    expand_44: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_27, [8, 16, 48, 784]);  div_27 = None
    clone_79: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_133: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_79, [128, 48, 784]);  clone_79 = None
    expand_45: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_68, [8, 16, 784, 48]);  permute_68 = None
    clone_80: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_134: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_80, [128, 784, 48]);  clone_80 = None
    bmm_14: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
    view_135: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_14, [8, 16, 48, 48]);  bmm_14 = None
    mul_156: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_135, arg29_1);  view_135 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_156, [-1], True)
    sub_39: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_156, amax_7);  mul_156 = amax_7 = None
    exp_7: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_24: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_29: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_7, sum_24);  exp_7 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_81: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_46: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_81, [8, 16, 48, 48]);  clone_81 = None
    view_136: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_46, [128, 48, 48]);  expand_46 = None
    expand_47: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_67, [8, 16, 48, 784]);  getitem_67 = None
    clone_82: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_137: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_82, [128, 48, 784]);  clone_82 = None
    bmm_15: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_136, view_137);  view_136 = view_137 = None
    view_138: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_15, [8, 16, 48, 784]);  bmm_15 = None
    permute_69: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_138, [0, 3, 1, 2]);  view_138 = None
    view_139: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_69, [8, 784, 768]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    clone_83: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_139, memory_format = torch.contiguous_format);  view_139 = None
    view_140: "f32[6272, 768]" = torch.ops.aten.view.default(clone_83, [6272, 768]);  clone_83 = None
    mm_7: "f32[6272, 768]" = torch.ops.aten.mm.default(view_140, permute_70);  view_140 = permute_70 = None
    view_141: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_7, [8, 784, 768]);  mm_7 = None
    add_114: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_141, arg257_1);  view_141 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_84: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_114);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_157: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg28_1, clone_84);  arg28_1 = clone_84 = None
    add_115: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_111, mul_157);  add_111 = mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_85: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 784, 1]" = var_mean_22[0]
    getitem_69: "f32[8, 784, 1]" = var_mean_22[1];  var_mean_22 = None
    add_116: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_22: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_40: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_85, getitem_69);  clone_85 = getitem_69 = None
    mul_158: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_22);  sub_40 = rsqrt_22 = None
    mul_159: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_158, arg258_1);  mul_158 = arg258_1 = None
    add_117: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_159, arg259_1);  mul_159 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_71: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_117, [0, 2, 1]);  add_117 = None
    view_142: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_71, [8, 768, 28, 28]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_18: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_142, arg260_1, arg261_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_142 = arg260_1 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_160: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_161: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476);  convolution_18 = None
    erf_16: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
    add_118: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_162: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_160, add_118);  mul_160 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_26: "f32[768]" = torch.ops.prims.convert_element_type.default(arg658_1, torch.float32);  arg658_1 = None
    convert_element_type_27: "f32[768]" = torch.ops.prims.convert_element_type.default(arg659_1, torch.float32);  arg659_1 = None
    add_119: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_10: "f32[768]" = torch.ops.aten.sqrt.default(add_119);  add_119 = None
    reciprocal_10: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_163: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_87: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_88: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, -1);  unsqueeze_87 = None
    unsqueeze_89: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_163, -1);  mul_163 = None
    unsqueeze_90: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, -1);  unsqueeze_89 = None
    sub_41: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_162, unsqueeze_88);  mul_162 = unsqueeze_88 = None
    mul_164: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_90);  sub_41 = unsqueeze_90 = None
    unsqueeze_91: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
    unsqueeze_92: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, -1);  unsqueeze_91 = None
    mul_165: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_92);  mul_164 = unsqueeze_92 = None
    unsqueeze_93: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
    unsqueeze_94: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, -1);  unsqueeze_93 = None
    add_120: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_165, unsqueeze_94);  mul_165 = unsqueeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_19: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_120, arg264_1, arg265_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_120 = arg264_1 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_143: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_19, [8, 768, 784]);  convolution_19 = None
    permute_72: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_166: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg30_1, permute_72);  arg30_1 = permute_72 = None
    add_121: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_115, mul_166);  add_115 = mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_86: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_121, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_86, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 784, 1]" = var_mean_23[0]
    getitem_71: "f32[8, 784, 1]" = var_mean_23[1];  var_mean_23 = None
    add_122: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_23: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_42: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_86, getitem_71);  clone_86 = getitem_71 = None
    mul_167: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_23);  sub_42 = rsqrt_23 = None
    mul_168: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_167, arg266_1);  mul_167 = arg266_1 = None
    add_123: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_168, arg267_1);  mul_168 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_144: "f32[6272, 768]" = torch.ops.aten.view.default(add_123, [6272, 768]);  add_123 = None
    permute_73: "f32[768, 3072]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
    addmm_22: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg269_1, view_144, permute_73);  arg269_1 = view_144 = permute_73 = None
    view_145: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 784, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_169: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_145, 0.5)
    mul_170: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_145, 0.7071067811865476);  view_145 = None
    erf_17: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_170);  mul_170 = None
    add_124: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_171: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_169, add_124);  mul_169 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_87: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_171);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_146: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_87, [6272, 3072]);  clone_87 = None
    permute_74: "f32[3072, 768]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_23: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg271_1, view_146, permute_74);  arg271_1 = view_146 = permute_74 = None
    view_147: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_23, [8, 784, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_88: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_147);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_172: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg31_1, clone_88);  arg31_1 = clone_88 = None
    add_125: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_121, mul_172);  add_121 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_89: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_89, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 784, 1]" = var_mean_24[0]
    getitem_73: "f32[8, 784, 1]" = var_mean_24[1];  var_mean_24 = None
    add_126: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_24: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_43: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_89, getitem_73);  clone_89 = getitem_73 = None
    mul_173: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_24);  sub_43 = rsqrt_24 = None
    mul_174: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_173, arg272_1);  mul_173 = arg272_1 = None
    add_127: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_174, arg273_1);  mul_174 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_148: "f32[6272, 768]" = torch.ops.aten.view.default(add_127, [6272, 768]);  add_127 = None
    permute_75: "f32[768, 2304]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
    addmm_24: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg275_1, view_148, permute_75);  arg275_1 = view_148 = permute_75 = None
    view_149: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 784, 2304]);  addmm_24 = None
    view_150: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_149, [8, 784, 3, 16, 48]);  view_149 = None
    permute_76: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_150, [2, 0, 3, 4, 1]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
    getitem_74: "f32[8, 16, 48, 784]" = unbind_8[0]
    getitem_75: "f32[8, 16, 48, 784]" = unbind_8[1]
    getitem_76: "f32[8, 16, 48, 784]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_34: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_74, 2.0)
    sum_25: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_34, [-1], True);  pow_34 = None
    pow_35: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_25, 0.5);  sum_25 = None
    clamp_min_16: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_35, 1e-12);  pow_35 = None
    expand_48: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_16, [8, 16, 48, 784]);  clamp_min_16 = None
    div_30: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_74, expand_48);  getitem_74 = expand_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_36: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_75, 2.0)
    sum_26: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_36, [-1], True);  pow_36 = None
    pow_37: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_26, 0.5);  sum_26 = None
    clamp_min_17: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_37, 1e-12);  pow_37 = None
    expand_49: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_17, [8, 16, 48, 784]);  clamp_min_17 = None
    div_31: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_75, expand_49);  getitem_75 = expand_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_77: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_31, [0, 1, 3, 2]);  div_31 = None
    expand_50: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_30, [8, 16, 48, 784]);  div_30 = None
    clone_90: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_151: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_90, [128, 48, 784]);  clone_90 = None
    expand_51: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_77, [8, 16, 784, 48]);  permute_77 = None
    clone_91: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_152: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_91, [128, 784, 48]);  clone_91 = None
    bmm_16: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_151, view_152);  view_151 = view_152 = None
    view_153: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_16, [8, 16, 48, 48]);  bmm_16 = None
    mul_175: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_153, arg33_1);  view_153 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_175, [-1], True)
    sub_44: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_175, amax_8);  mul_175 = amax_8 = None
    exp_8: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_27: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_32: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_8, sum_27);  exp_8 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_92: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_52: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_92, [8, 16, 48, 48]);  clone_92 = None
    view_154: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_52, [128, 48, 48]);  expand_52 = None
    expand_53: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_76, [8, 16, 48, 784]);  getitem_76 = None
    clone_93: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_155: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_93, [128, 48, 784]);  clone_93 = None
    bmm_17: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_154, view_155);  view_154 = view_155 = None
    view_156: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_17, [8, 16, 48, 784]);  bmm_17 = None
    permute_78: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_156, [0, 3, 1, 2]);  view_156 = None
    view_157: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_78, [8, 784, 768]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    clone_94: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_157, memory_format = torch.contiguous_format);  view_157 = None
    view_158: "f32[6272, 768]" = torch.ops.aten.view.default(clone_94, [6272, 768]);  clone_94 = None
    mm_8: "f32[6272, 768]" = torch.ops.aten.mm.default(view_158, permute_79);  view_158 = permute_79 = None
    view_159: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_8, [8, 784, 768]);  mm_8 = None
    add_128: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_159, arg277_1);  view_159 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_95: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_176: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg32_1, clone_95);  arg32_1 = clone_95 = None
    add_129: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_125, mul_176);  add_125 = mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_96: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 784, 1]" = var_mean_25[0]
    getitem_78: "f32[8, 784, 1]" = var_mean_25[1];  var_mean_25 = None
    add_130: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_25: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_45: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_78);  clone_96 = getitem_78 = None
    mul_177: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_25);  sub_45 = rsqrt_25 = None
    mul_178: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_177, arg278_1);  mul_177 = arg278_1 = None
    add_131: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_178, arg279_1);  mul_178 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_80: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_131, [0, 2, 1]);  add_131 = None
    view_160: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_80, [8, 768, 28, 28]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_20: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_160, arg280_1, arg281_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_160 = arg280_1 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_179: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_180: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476);  convolution_20 = None
    erf_18: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
    add_132: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_181: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_179, add_132);  mul_179 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_28: "f32[768]" = torch.ops.prims.convert_element_type.default(arg661_1, torch.float32);  arg661_1 = None
    convert_element_type_29: "f32[768]" = torch.ops.prims.convert_element_type.default(arg662_1, torch.float32);  arg662_1 = None
    add_133: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_11: "f32[768]" = torch.ops.aten.sqrt.default(add_133);  add_133 = None
    reciprocal_11: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_182: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_95: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_96: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, -1);  unsqueeze_95 = None
    unsqueeze_97: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
    unsqueeze_98: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, -1);  unsqueeze_97 = None
    sub_46: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_181, unsqueeze_96);  mul_181 = unsqueeze_96 = None
    mul_183: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_98);  sub_46 = unsqueeze_98 = None
    unsqueeze_99: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
    unsqueeze_100: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, -1);  unsqueeze_99 = None
    mul_184: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_100);  mul_183 = unsqueeze_100 = None
    unsqueeze_101: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
    unsqueeze_102: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, -1);  unsqueeze_101 = None
    add_134: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_102);  mul_184 = unsqueeze_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_21: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_134, arg284_1, arg285_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_134 = arg284_1 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_161: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_21, [8, 768, 784]);  convolution_21 = None
    permute_81: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_185: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg34_1, permute_81);  arg34_1 = permute_81 = None
    add_135: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_129, mul_185);  add_129 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_97: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_135, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
    getitem_79: "f32[8, 784, 1]" = var_mean_26[0]
    getitem_80: "f32[8, 784, 1]" = var_mean_26[1];  var_mean_26 = None
    add_136: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_79, 1e-06);  getitem_79 = None
    rsqrt_26: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_47: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_97, getitem_80);  clone_97 = getitem_80 = None
    mul_186: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_26);  sub_47 = rsqrt_26 = None
    mul_187: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_186, arg286_1);  mul_186 = arg286_1 = None
    add_137: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_187, arg287_1);  mul_187 = arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_162: "f32[6272, 768]" = torch.ops.aten.view.default(add_137, [6272, 768]);  add_137 = None
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
    addmm_25: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg289_1, view_162, permute_82);  arg289_1 = view_162 = permute_82 = None
    view_163: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_25, [8, 784, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_188: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.5)
    mul_189: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.7071067811865476);  view_163 = None
    erf_19: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_189);  mul_189 = None
    add_138: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_190: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_188, add_138);  mul_188 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_98: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_190);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_164: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_98, [6272, 3072]);  clone_98 = None
    permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
    addmm_26: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg291_1, view_164, permute_83);  arg291_1 = view_164 = permute_83 = None
    view_165: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_26, [8, 784, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_99: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_165);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_191: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg35_1, clone_99);  arg35_1 = clone_99 = None
    add_139: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_135, mul_191);  add_135 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_100: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_100, [2], correction = 0, keepdim = True)
    getitem_81: "f32[8, 784, 1]" = var_mean_27[0]
    getitem_82: "f32[8, 784, 1]" = var_mean_27[1];  var_mean_27 = None
    add_140: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_81, 1e-06);  getitem_81 = None
    rsqrt_27: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_48: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_100, getitem_82);  clone_100 = getitem_82 = None
    mul_192: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_27);  sub_48 = rsqrt_27 = None
    mul_193: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_192, arg292_1);  mul_192 = arg292_1 = None
    add_141: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_193, arg293_1);  mul_193 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_166: "f32[6272, 768]" = torch.ops.aten.view.default(add_141, [6272, 768]);  add_141 = None
    permute_84: "f32[768, 2304]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
    addmm_27: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg295_1, view_166, permute_84);  arg295_1 = view_166 = permute_84 = None
    view_167: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_27, [8, 784, 2304]);  addmm_27 = None
    view_168: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_167, [8, 784, 3, 16, 48]);  view_167 = None
    permute_85: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_168, [2, 0, 3, 4, 1]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_85);  permute_85 = None
    getitem_83: "f32[8, 16, 48, 784]" = unbind_9[0]
    getitem_84: "f32[8, 16, 48, 784]" = unbind_9[1]
    getitem_85: "f32[8, 16, 48, 784]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_38: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_83, 2.0)
    sum_28: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_38, [-1], True);  pow_38 = None
    pow_39: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_28, 0.5);  sum_28 = None
    clamp_min_18: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_39, 1e-12);  pow_39 = None
    expand_54: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_18, [8, 16, 48, 784]);  clamp_min_18 = None
    div_33: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_83, expand_54);  getitem_83 = expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_40: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_84, 2.0)
    sum_29: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_40, [-1], True);  pow_40 = None
    pow_41: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_29, 0.5);  sum_29 = None
    clamp_min_19: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_41, 1e-12);  pow_41 = None
    expand_55: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_19, [8, 16, 48, 784]);  clamp_min_19 = None
    div_34: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_84, expand_55);  getitem_84 = expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_86: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_34, [0, 1, 3, 2]);  div_34 = None
    expand_56: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_33, [8, 16, 48, 784]);  div_33 = None
    clone_101: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_169: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_101, [128, 48, 784]);  clone_101 = None
    expand_57: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_86, [8, 16, 784, 48]);  permute_86 = None
    clone_102: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_170: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_102, [128, 784, 48]);  clone_102 = None
    bmm_18: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_169, view_170);  view_169 = view_170 = None
    view_171: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_18, [8, 16, 48, 48]);  bmm_18 = None
    mul_194: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_171, arg37_1);  view_171 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_194, [-1], True)
    sub_49: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_194, amax_9);  mul_194 = amax_9 = None
    exp_9: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_30: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_35: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_9, sum_30);  exp_9 = sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_103: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_58: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_103, [8, 16, 48, 48]);  clone_103 = None
    view_172: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_58, [128, 48, 48]);  expand_58 = None
    expand_59: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_85, [8, 16, 48, 784]);  getitem_85 = None
    clone_104: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_173: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_104, [128, 48, 784]);  clone_104 = None
    bmm_19: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_172, view_173);  view_172 = view_173 = None
    view_174: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_19, [8, 16, 48, 784]);  bmm_19 = None
    permute_87: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_174, [0, 3, 1, 2]);  view_174 = None
    view_175: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_87, [8, 784, 768]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
    clone_105: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_175, memory_format = torch.contiguous_format);  view_175 = None
    view_176: "f32[6272, 768]" = torch.ops.aten.view.default(clone_105, [6272, 768]);  clone_105 = None
    mm_9: "f32[6272, 768]" = torch.ops.aten.mm.default(view_176, permute_88);  view_176 = permute_88 = None
    view_177: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_9, [8, 784, 768]);  mm_9 = None
    add_142: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_177, arg297_1);  view_177 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_106: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_142);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_195: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg36_1, clone_106);  arg36_1 = clone_106 = None
    add_143: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_139, mul_195);  add_139 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_107: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_107, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 784, 1]" = var_mean_28[0]
    getitem_87: "f32[8, 784, 1]" = var_mean_28[1];  var_mean_28 = None
    add_144: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_28: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_50: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_107, getitem_87);  clone_107 = getitem_87 = None
    mul_196: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_28);  sub_50 = rsqrt_28 = None
    mul_197: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_196, arg298_1);  mul_196 = arg298_1 = None
    add_145: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_197, arg299_1);  mul_197 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_89: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_145, [0, 2, 1]);  add_145 = None
    view_178: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_89, [8, 768, 28, 28]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_22: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_178, arg300_1, arg301_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_178 = arg300_1 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_198: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.5)
    mul_199: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476);  convolution_22 = None
    erf_20: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_199);  mul_199 = None
    add_146: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_200: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_198, add_146);  mul_198 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_30: "f32[768]" = torch.ops.prims.convert_element_type.default(arg664_1, torch.float32);  arg664_1 = None
    convert_element_type_31: "f32[768]" = torch.ops.prims.convert_element_type.default(arg665_1, torch.float32);  arg665_1 = None
    add_147: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_12: "f32[768]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
    reciprocal_12: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_201: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_103: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_104: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, -1);  unsqueeze_103 = None
    unsqueeze_105: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
    unsqueeze_106: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, -1);  unsqueeze_105 = None
    sub_51: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_200, unsqueeze_104);  mul_200 = unsqueeze_104 = None
    mul_202: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_106);  sub_51 = unsqueeze_106 = None
    unsqueeze_107: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
    unsqueeze_108: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, -1);  unsqueeze_107 = None
    mul_203: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_108);  mul_202 = unsqueeze_108 = None
    unsqueeze_109: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
    unsqueeze_110: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, -1);  unsqueeze_109 = None
    add_148: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_110);  mul_203 = unsqueeze_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_23: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_148, arg304_1, arg305_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_148 = arg304_1 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_179: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_23, [8, 768, 784]);  convolution_23 = None
    permute_90: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_204: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg38_1, permute_90);  arg38_1 = permute_90 = None
    add_149: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_143, mul_204);  add_143 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_108: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_149, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_108, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 784, 1]" = var_mean_29[0]
    getitem_89: "f32[8, 784, 1]" = var_mean_29[1];  var_mean_29 = None
    add_150: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_29: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_52: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_108, getitem_89);  clone_108 = getitem_89 = None
    mul_205: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_29);  sub_52 = rsqrt_29 = None
    mul_206: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_205, arg306_1);  mul_205 = arg306_1 = None
    add_151: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_206, arg307_1);  mul_206 = arg307_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[6272, 768]" = torch.ops.aten.view.default(add_151, [6272, 768]);  add_151 = None
    permute_91: "f32[768, 3072]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
    addmm_28: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg309_1, view_180, permute_91);  arg309_1 = view_180 = permute_91 = None
    view_181: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_28, [8, 784, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_207: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_208: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
    erf_21: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_152: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_209: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_207, add_152);  mul_207 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_109: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_209);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_109, [6272, 3072]);  clone_109 = None
    permute_92: "f32[3072, 768]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
    addmm_29: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg311_1, view_182, permute_92);  arg311_1 = view_182 = permute_92 = None
    view_183: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_29, [8, 784, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_110: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_183);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_210: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg39_1, clone_110);  arg39_1 = clone_110 = None
    add_153: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_149, mul_210);  add_149 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_111: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_111, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 784, 1]" = var_mean_30[0]
    getitem_91: "f32[8, 784, 1]" = var_mean_30[1];  var_mean_30 = None
    add_154: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_30: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_53: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_111, getitem_91);  clone_111 = getitem_91 = None
    mul_211: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_30);  sub_53 = rsqrt_30 = None
    mul_212: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_211, arg312_1);  mul_211 = arg312_1 = None
    add_155: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_212, arg313_1);  mul_212 = arg313_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_184: "f32[6272, 768]" = torch.ops.aten.view.default(add_155, [6272, 768]);  add_155 = None
    permute_93: "f32[768, 2304]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
    addmm_30: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg315_1, view_184, permute_93);  arg315_1 = view_184 = permute_93 = None
    view_185: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_30, [8, 784, 2304]);  addmm_30 = None
    view_186: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_185, [8, 784, 3, 16, 48]);  view_185 = None
    permute_94: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 4, 1]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_94);  permute_94 = None
    getitem_92: "f32[8, 16, 48, 784]" = unbind_10[0]
    getitem_93: "f32[8, 16, 48, 784]" = unbind_10[1]
    getitem_94: "f32[8, 16, 48, 784]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_42: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_92, 2.0)
    sum_31: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_42, [-1], True);  pow_42 = None
    pow_43: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_31, 0.5);  sum_31 = None
    clamp_min_20: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_43, 1e-12);  pow_43 = None
    expand_60: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_20, [8, 16, 48, 784]);  clamp_min_20 = None
    div_36: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_92, expand_60);  getitem_92 = expand_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_44: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_93, 2.0)
    sum_32: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_44, [-1], True);  pow_44 = None
    pow_45: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_32, 0.5);  sum_32 = None
    clamp_min_21: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_45, 1e-12);  pow_45 = None
    expand_61: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_21, [8, 16, 48, 784]);  clamp_min_21 = None
    div_37: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_93, expand_61);  getitem_93 = expand_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_95: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_37, [0, 1, 3, 2]);  div_37 = None
    expand_62: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_36, [8, 16, 48, 784]);  div_36 = None
    clone_112: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    view_187: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_112, [128, 48, 784]);  clone_112 = None
    expand_63: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_95, [8, 16, 784, 48]);  permute_95 = None
    clone_113: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_188: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_113, [128, 784, 48]);  clone_113 = None
    bmm_20: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
    view_189: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_20, [8, 16, 48, 48]);  bmm_20 = None
    mul_213: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_189, arg41_1);  view_189 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_213, [-1], True)
    sub_54: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_213, amax_10);  mul_213 = amax_10 = None
    exp_10: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_33: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_38: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_10, sum_33);  exp_10 = sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_114: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_64: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_114, [8, 16, 48, 48]);  clone_114 = None
    view_190: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_64, [128, 48, 48]);  expand_64 = None
    expand_65: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_94, [8, 16, 48, 784]);  getitem_94 = None
    clone_115: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_191: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_115, [128, 48, 784]);  clone_115 = None
    bmm_21: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_21, [8, 16, 48, 784]);  bmm_21 = None
    permute_96: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_192, [0, 3, 1, 2]);  view_192 = None
    view_193: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_96, [8, 784, 768]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
    clone_116: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_193, memory_format = torch.contiguous_format);  view_193 = None
    view_194: "f32[6272, 768]" = torch.ops.aten.view.default(clone_116, [6272, 768]);  clone_116 = None
    mm_10: "f32[6272, 768]" = torch.ops.aten.mm.default(view_194, permute_97);  view_194 = permute_97 = None
    view_195: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_10, [8, 784, 768]);  mm_10 = None
    add_156: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_195, arg317_1);  view_195 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_117: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_156);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_214: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg40_1, clone_117);  arg40_1 = clone_117 = None
    add_157: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_153, mul_214);  add_153 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_118: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_118, [2], correction = 0, keepdim = True)
    getitem_95: "f32[8, 784, 1]" = var_mean_31[0]
    getitem_96: "f32[8, 784, 1]" = var_mean_31[1];  var_mean_31 = None
    add_158: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_95, 1e-06);  getitem_95 = None
    rsqrt_31: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_55: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_118, getitem_96);  clone_118 = getitem_96 = None
    mul_215: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_31);  sub_55 = rsqrt_31 = None
    mul_216: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_215, arg318_1);  mul_215 = arg318_1 = None
    add_159: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_216, arg319_1);  mul_216 = arg319_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_98: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_159, [0, 2, 1]);  add_159 = None
    view_196: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_98, [8, 768, 28, 28]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_24: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_196, arg320_1, arg321_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_196 = arg320_1 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_217: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.5)
    mul_218: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.7071067811865476);  convolution_24 = None
    erf_22: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
    add_160: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_219: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_217, add_160);  mul_217 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_32: "f32[768]" = torch.ops.prims.convert_element_type.default(arg667_1, torch.float32);  arg667_1 = None
    convert_element_type_33: "f32[768]" = torch.ops.prims.convert_element_type.default(arg668_1, torch.float32);  arg668_1 = None
    add_161: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_13: "f32[768]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
    reciprocal_13: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_220: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_111: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_112: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, -1);  unsqueeze_111 = None
    unsqueeze_113: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_220, -1);  mul_220 = None
    unsqueeze_114: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, -1);  unsqueeze_113 = None
    sub_56: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_219, unsqueeze_112);  mul_219 = unsqueeze_112 = None
    mul_221: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_114);  sub_56 = unsqueeze_114 = None
    unsqueeze_115: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
    unsqueeze_116: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, -1);  unsqueeze_115 = None
    mul_222: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_116);  mul_221 = unsqueeze_116 = None
    unsqueeze_117: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
    unsqueeze_118: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, -1);  unsqueeze_117 = None
    add_162: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_118);  mul_222 = unsqueeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_25: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_162, arg324_1, arg325_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_162 = arg324_1 = arg325_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_197: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_25, [8, 768, 784]);  convolution_25 = None
    permute_99: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_223: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg42_1, permute_99);  arg42_1 = permute_99 = None
    add_163: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_157, mul_223);  add_157 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_119: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_163, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_119, [2], correction = 0, keepdim = True)
    getitem_97: "f32[8, 784, 1]" = var_mean_32[0]
    getitem_98: "f32[8, 784, 1]" = var_mean_32[1];  var_mean_32 = None
    add_164: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_32: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_57: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_119, getitem_98);  clone_119 = getitem_98 = None
    mul_224: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_32);  sub_57 = rsqrt_32 = None
    mul_225: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_224, arg326_1);  mul_224 = arg326_1 = None
    add_165: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_225, arg327_1);  mul_225 = arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[6272, 768]" = torch.ops.aten.view.default(add_165, [6272, 768]);  add_165 = None
    permute_100: "f32[768, 3072]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
    addmm_31: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg329_1, view_198, permute_100);  arg329_1 = view_198 = permute_100 = None
    view_199: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_31, [8, 784, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_226: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
    mul_227: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476);  view_199 = None
    erf_23: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_166: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_228: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_226, add_166);  mul_226 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_120: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_228);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_200: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_120, [6272, 3072]);  clone_120 = None
    permute_101: "f32[3072, 768]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
    addmm_32: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg331_1, view_200, permute_101);  arg331_1 = view_200 = permute_101 = None
    view_201: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_32, [8, 784, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_121: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_201);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_229: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg43_1, clone_121);  arg43_1 = clone_121 = None
    add_167: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_163, mul_229);  add_163 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_122: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_122, [2], correction = 0, keepdim = True)
    getitem_99: "f32[8, 784, 1]" = var_mean_33[0]
    getitem_100: "f32[8, 784, 1]" = var_mean_33[1];  var_mean_33 = None
    add_168: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_33: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_58: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_122, getitem_100);  clone_122 = getitem_100 = None
    mul_230: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_33);  sub_58 = rsqrt_33 = None
    mul_231: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_230, arg332_1);  mul_230 = arg332_1 = None
    add_169: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_231, arg333_1);  mul_231 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_202: "f32[6272, 768]" = torch.ops.aten.view.default(add_169, [6272, 768]);  add_169 = None
    permute_102: "f32[768, 2304]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
    addmm_33: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg335_1, view_202, permute_102);  arg335_1 = view_202 = permute_102 = None
    view_203: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_33, [8, 784, 2304]);  addmm_33 = None
    view_204: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_203, [8, 784, 3, 16, 48]);  view_203 = None
    permute_103: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_204, [2, 0, 3, 4, 1]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_103);  permute_103 = None
    getitem_101: "f32[8, 16, 48, 784]" = unbind_11[0]
    getitem_102: "f32[8, 16, 48, 784]" = unbind_11[1]
    getitem_103: "f32[8, 16, 48, 784]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_46: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_101, 2.0)
    sum_34: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_46, [-1], True);  pow_46 = None
    pow_47: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_34, 0.5);  sum_34 = None
    clamp_min_22: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_47, 1e-12);  pow_47 = None
    expand_66: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_22, [8, 16, 48, 784]);  clamp_min_22 = None
    div_39: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_101, expand_66);  getitem_101 = expand_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_48: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_102, 2.0)
    sum_35: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_48, [-1], True);  pow_48 = None
    pow_49: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_35, 0.5);  sum_35 = None
    clamp_min_23: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_49, 1e-12);  pow_49 = None
    expand_67: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_23, [8, 16, 48, 784]);  clamp_min_23 = None
    div_40: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_102, expand_67);  getitem_102 = expand_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_104: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_40, [0, 1, 3, 2]);  div_40 = None
    expand_68: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_39, [8, 16, 48, 784]);  div_39 = None
    clone_123: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_205: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_123, [128, 48, 784]);  clone_123 = None
    expand_69: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_104, [8, 16, 784, 48]);  permute_104 = None
    clone_124: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_206: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_124, [128, 784, 48]);  clone_124 = None
    bmm_22: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_205, view_206);  view_205 = view_206 = None
    view_207: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_22, [8, 16, 48, 48]);  bmm_22 = None
    mul_232: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_207, arg45_1);  view_207 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_232, [-1], True)
    sub_59: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_232, amax_11);  mul_232 = amax_11 = None
    exp_11: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_36: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_41: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_11, sum_36);  exp_11 = sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_125: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_70: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_125, [8, 16, 48, 48]);  clone_125 = None
    view_208: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_70, [128, 48, 48]);  expand_70 = None
    expand_71: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_103, [8, 16, 48, 784]);  getitem_103 = None
    clone_126: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_209: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_126, [128, 48, 784]);  clone_126 = None
    bmm_23: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_208, view_209);  view_208 = view_209 = None
    view_210: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_23, [8, 16, 48, 784]);  bmm_23 = None
    permute_105: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_210, [0, 3, 1, 2]);  view_210 = None
    view_211: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_105, [8, 784, 768]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
    clone_127: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_211, memory_format = torch.contiguous_format);  view_211 = None
    view_212: "f32[6272, 768]" = torch.ops.aten.view.default(clone_127, [6272, 768]);  clone_127 = None
    mm_11: "f32[6272, 768]" = torch.ops.aten.mm.default(view_212, permute_106);  view_212 = permute_106 = None
    view_213: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_11, [8, 784, 768]);  mm_11 = None
    add_170: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_213, arg337_1);  view_213 = arg337_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_128: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_170);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_233: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg44_1, clone_128);  arg44_1 = clone_128 = None
    add_171: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_167, mul_233);  add_167 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_129: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_129, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 784, 1]" = var_mean_34[0]
    getitem_105: "f32[8, 784, 1]" = var_mean_34[1];  var_mean_34 = None
    add_172: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_34: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_60: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_129, getitem_105);  clone_129 = getitem_105 = None
    mul_234: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_34);  sub_60 = rsqrt_34 = None
    mul_235: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_234, arg338_1);  mul_234 = arg338_1 = None
    add_173: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_235, arg339_1);  mul_235 = arg339_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_107: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_173, [0, 2, 1]);  add_173 = None
    view_214: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_107, [8, 768, 28, 28]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_26: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_214, arg340_1, arg341_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_214 = arg340_1 = arg341_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_236: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, 0.5)
    mul_237: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476);  convolution_26 = None
    erf_24: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_237);  mul_237 = None
    add_174: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_238: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_236, add_174);  mul_236 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_34: "f32[768]" = torch.ops.prims.convert_element_type.default(arg670_1, torch.float32);  arg670_1 = None
    convert_element_type_35: "f32[768]" = torch.ops.prims.convert_element_type.default(arg671_1, torch.float32);  arg671_1 = None
    add_175: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_14: "f32[768]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
    reciprocal_14: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_239: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_119: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_120: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, -1);  unsqueeze_119 = None
    unsqueeze_121: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_239, -1);  mul_239 = None
    unsqueeze_122: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, -1);  unsqueeze_121 = None
    sub_61: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_238, unsqueeze_120);  mul_238 = unsqueeze_120 = None
    mul_240: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_122);  sub_61 = unsqueeze_122 = None
    unsqueeze_123: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
    unsqueeze_124: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, -1);  unsqueeze_123 = None
    mul_241: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_124);  mul_240 = unsqueeze_124 = None
    unsqueeze_125: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
    unsqueeze_126: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, -1);  unsqueeze_125 = None
    add_176: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_241, unsqueeze_126);  mul_241 = unsqueeze_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_27: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_176, arg344_1, arg345_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_176 = arg344_1 = arg345_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_215: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_27, [8, 768, 784]);  convolution_27 = None
    permute_108: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_242: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg46_1, permute_108);  arg46_1 = permute_108 = None
    add_177: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_171, mul_242);  add_171 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_130: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_177, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 784, 1]" = var_mean_35[0]
    getitem_107: "f32[8, 784, 1]" = var_mean_35[1];  var_mean_35 = None
    add_178: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_35: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_62: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_130, getitem_107);  clone_130 = getitem_107 = None
    mul_243: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_35);  sub_62 = rsqrt_35 = None
    mul_244: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_243, arg346_1);  mul_243 = arg346_1 = None
    add_179: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_244, arg347_1);  mul_244 = arg347_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_216: "f32[6272, 768]" = torch.ops.aten.view.default(add_179, [6272, 768]);  add_179 = None
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
    addmm_34: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg349_1, view_216, permute_109);  arg349_1 = view_216 = permute_109 = None
    view_217: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 784, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_245: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_246: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_25: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
    add_180: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_247: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_245, add_180);  mul_245 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_131: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_247);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_218: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_131, [6272, 3072]);  clone_131 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
    addmm_35: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg351_1, view_218, permute_110);  arg351_1 = view_218 = permute_110 = None
    view_219: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_35, [8, 784, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_132: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_219);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_248: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg47_1, clone_132);  arg47_1 = clone_132 = None
    add_181: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_177, mul_248);  add_177 = mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_133: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_181, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 784, 1]" = var_mean_36[0]
    getitem_109: "f32[8, 784, 1]" = var_mean_36[1];  var_mean_36 = None
    add_182: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_36: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_63: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_133, getitem_109);  clone_133 = getitem_109 = None
    mul_249: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_36);  sub_63 = rsqrt_36 = None
    mul_250: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_249, arg352_1);  mul_249 = arg352_1 = None
    add_183: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_250, arg353_1);  mul_250 = arg353_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_220: "f32[6272, 768]" = torch.ops.aten.view.default(add_183, [6272, 768]);  add_183 = None
    permute_111: "f32[768, 2304]" = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
    addmm_36: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg355_1, view_220, permute_111);  arg355_1 = view_220 = permute_111 = None
    view_221: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 784, 2304]);  addmm_36 = None
    view_222: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_221, [8, 784, 3, 16, 48]);  view_221 = None
    permute_112: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_222, [2, 0, 3, 4, 1]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
    getitem_110: "f32[8, 16, 48, 784]" = unbind_12[0]
    getitem_111: "f32[8, 16, 48, 784]" = unbind_12[1]
    getitem_112: "f32[8, 16, 48, 784]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_50: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_110, 2.0)
    sum_37: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_50, [-1], True);  pow_50 = None
    pow_51: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_37, 0.5);  sum_37 = None
    clamp_min_24: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_51, 1e-12);  pow_51 = None
    expand_72: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_24, [8, 16, 48, 784]);  clamp_min_24 = None
    div_42: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_110, expand_72);  getitem_110 = expand_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_52: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_111, 2.0)
    sum_38: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_52, [-1], True);  pow_52 = None
    pow_53: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_38, 0.5);  sum_38 = None
    clamp_min_25: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_53, 1e-12);  pow_53 = None
    expand_73: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_25, [8, 16, 48, 784]);  clamp_min_25 = None
    div_43: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_111, expand_73);  getitem_111 = expand_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_113: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_43, [0, 1, 3, 2]);  div_43 = None
    expand_74: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_42, [8, 16, 48, 784]);  div_42 = None
    clone_134: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_223: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_134, [128, 48, 784]);  clone_134 = None
    expand_75: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_113, [8, 16, 784, 48]);  permute_113 = None
    clone_135: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_224: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_135, [128, 784, 48]);  clone_135 = None
    bmm_24: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_223, view_224);  view_223 = view_224 = None
    view_225: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_24, [8, 16, 48, 48]);  bmm_24 = None
    mul_251: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_225, arg49_1);  view_225 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_251, [-1], True)
    sub_64: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_251, amax_12);  mul_251 = amax_12 = None
    exp_12: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_39: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_44: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_12, sum_39);  exp_12 = sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_136: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_76: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_136, [8, 16, 48, 48]);  clone_136 = None
    view_226: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_76, [128, 48, 48]);  expand_76 = None
    expand_77: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_112, [8, 16, 48, 784]);  getitem_112 = None
    clone_137: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_227: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_137, [128, 48, 784]);  clone_137 = None
    bmm_25: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_226, view_227);  view_226 = view_227 = None
    view_228: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_25, [8, 16, 48, 784]);  bmm_25 = None
    permute_114: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_228, [0, 3, 1, 2]);  view_228 = None
    view_229: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_114, [8, 784, 768]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
    clone_138: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_229, memory_format = torch.contiguous_format);  view_229 = None
    view_230: "f32[6272, 768]" = torch.ops.aten.view.default(clone_138, [6272, 768]);  clone_138 = None
    mm_12: "f32[6272, 768]" = torch.ops.aten.mm.default(view_230, permute_115);  view_230 = permute_115 = None
    view_231: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_12, [8, 784, 768]);  mm_12 = None
    add_184: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_231, arg357_1);  view_231 = arg357_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_139: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_252: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg48_1, clone_139);  arg48_1 = clone_139 = None
    add_185: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_181, mul_252);  add_181 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_140: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_185, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_113: "f32[8, 784, 1]" = var_mean_37[0]
    getitem_114: "f32[8, 784, 1]" = var_mean_37[1];  var_mean_37 = None
    add_186: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_113, 1e-06);  getitem_113 = None
    rsqrt_37: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_65: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_140, getitem_114);  clone_140 = getitem_114 = None
    mul_253: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_37);  sub_65 = rsqrt_37 = None
    mul_254: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_253, arg358_1);  mul_253 = arg358_1 = None
    add_187: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_254, arg359_1);  mul_254 = arg359_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_116: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_187, [0, 2, 1]);  add_187 = None
    view_232: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_116, [8, 768, 28, 28]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_28: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_232, arg360_1, arg361_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_232 = arg360_1 = arg361_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_255: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, 0.5)
    mul_256: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, 0.7071067811865476);  convolution_28 = None
    erf_26: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_256);  mul_256 = None
    add_188: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_257: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_255, add_188);  mul_255 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_36: "f32[768]" = torch.ops.prims.convert_element_type.default(arg673_1, torch.float32);  arg673_1 = None
    convert_element_type_37: "f32[768]" = torch.ops.prims.convert_element_type.default(arg674_1, torch.float32);  arg674_1 = None
    add_189: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_15: "f32[768]" = torch.ops.aten.sqrt.default(add_189);  add_189 = None
    reciprocal_15: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_258: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_127: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_128: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, -1);  unsqueeze_127 = None
    unsqueeze_129: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
    unsqueeze_130: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, -1);  unsqueeze_129 = None
    sub_66: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_257, unsqueeze_128);  mul_257 = unsqueeze_128 = None
    mul_259: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_130);  sub_66 = unsqueeze_130 = None
    unsqueeze_131: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
    unsqueeze_132: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, -1);  unsqueeze_131 = None
    mul_260: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_132);  mul_259 = unsqueeze_132 = None
    unsqueeze_133: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg363_1, -1);  arg363_1 = None
    unsqueeze_134: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, -1);  unsqueeze_133 = None
    add_190: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_134);  mul_260 = unsqueeze_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_29: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_190, arg364_1, arg365_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_190 = arg364_1 = arg365_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_233: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_29, [8, 768, 784]);  convolution_29 = None
    permute_117: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_261: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg50_1, permute_117);  arg50_1 = permute_117 = None
    add_191: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_185, mul_261);  add_185 = mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_141: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_191, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_141, [2], correction = 0, keepdim = True)
    getitem_115: "f32[8, 784, 1]" = var_mean_38[0]
    getitem_116: "f32[8, 784, 1]" = var_mean_38[1];  var_mean_38 = None
    add_192: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_115, 1e-06);  getitem_115 = None
    rsqrt_38: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_67: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_141, getitem_116);  clone_141 = getitem_116 = None
    mul_262: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_38);  sub_67 = rsqrt_38 = None
    mul_263: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_262, arg366_1);  mul_262 = arg366_1 = None
    add_193: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_263, arg367_1);  mul_263 = arg367_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_234: "f32[6272, 768]" = torch.ops.aten.view.default(add_193, [6272, 768]);  add_193 = None
    permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
    addmm_37: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg369_1, view_234, permute_118);  arg369_1 = view_234 = permute_118 = None
    view_235: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_37, [8, 784, 3072]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_264: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_235, 0.5)
    mul_265: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_235, 0.7071067811865476);  view_235 = None
    erf_27: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_265);  mul_265 = None
    add_194: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_266: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_264, add_194);  mul_264 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_142: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_266);  mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_236: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_142, [6272, 3072]);  clone_142 = None
    permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
    addmm_38: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg371_1, view_236, permute_119);  arg371_1 = view_236 = permute_119 = None
    view_237: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_38, [8, 784, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_143: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_237);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_267: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg51_1, clone_143);  arg51_1 = clone_143 = None
    add_195: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_191, mul_267);  add_191 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_144: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_195, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_117: "f32[8, 784, 1]" = var_mean_39[0]
    getitem_118: "f32[8, 784, 1]" = var_mean_39[1];  var_mean_39 = None
    add_196: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_117, 1e-06);  getitem_117 = None
    rsqrt_39: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_68: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_144, getitem_118);  clone_144 = getitem_118 = None
    mul_268: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_39);  sub_68 = rsqrt_39 = None
    mul_269: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_268, arg372_1);  mul_268 = arg372_1 = None
    add_197: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_269, arg373_1);  mul_269 = arg373_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_238: "f32[6272, 768]" = torch.ops.aten.view.default(add_197, [6272, 768]);  add_197 = None
    permute_120: "f32[768, 2304]" = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
    addmm_39: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg375_1, view_238, permute_120);  arg375_1 = view_238 = permute_120 = None
    view_239: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_39, [8, 784, 2304]);  addmm_39 = None
    view_240: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_239, [8, 784, 3, 16, 48]);  view_239 = None
    permute_121: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_240, [2, 0, 3, 4, 1]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_121);  permute_121 = None
    getitem_119: "f32[8, 16, 48, 784]" = unbind_13[0]
    getitem_120: "f32[8, 16, 48, 784]" = unbind_13[1]
    getitem_121: "f32[8, 16, 48, 784]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_54: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_119, 2.0)
    sum_40: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_54, [-1], True);  pow_54 = None
    pow_55: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_40, 0.5);  sum_40 = None
    clamp_min_26: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_55, 1e-12);  pow_55 = None
    expand_78: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_26, [8, 16, 48, 784]);  clamp_min_26 = None
    div_45: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_119, expand_78);  getitem_119 = expand_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_56: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_120, 2.0)
    sum_41: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_56, [-1], True);  pow_56 = None
    pow_57: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_41, 0.5);  sum_41 = None
    clamp_min_27: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_57, 1e-12);  pow_57 = None
    expand_79: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_27, [8, 16, 48, 784]);  clamp_min_27 = None
    div_46: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_120, expand_79);  getitem_120 = expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_122: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_46, [0, 1, 3, 2]);  div_46 = None
    expand_80: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_45, [8, 16, 48, 784]);  div_45 = None
    clone_145: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_241: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_145, [128, 48, 784]);  clone_145 = None
    expand_81: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_122, [8, 16, 784, 48]);  permute_122 = None
    clone_146: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_242: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_146, [128, 784, 48]);  clone_146 = None
    bmm_26: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_241, view_242);  view_241 = view_242 = None
    view_243: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_26, [8, 16, 48, 48]);  bmm_26 = None
    mul_270: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_243, arg53_1);  view_243 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_270, [-1], True)
    sub_69: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_270, amax_13);  mul_270 = amax_13 = None
    exp_13: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_42: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_47: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_13, sum_42);  exp_13 = sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_147: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_47);  div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_82: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_147, [8, 16, 48, 48]);  clone_147 = None
    view_244: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_82, [128, 48, 48]);  expand_82 = None
    expand_83: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_121, [8, 16, 48, 784]);  getitem_121 = None
    clone_148: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_245: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_148, [128, 48, 784]);  clone_148 = None
    bmm_27: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
    view_246: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_27, [8, 16, 48, 784]);  bmm_27 = None
    permute_123: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_246, [0, 3, 1, 2]);  view_246 = None
    view_247: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_123, [8, 784, 768]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
    clone_149: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_247, memory_format = torch.contiguous_format);  view_247 = None
    view_248: "f32[6272, 768]" = torch.ops.aten.view.default(clone_149, [6272, 768]);  clone_149 = None
    mm_13: "f32[6272, 768]" = torch.ops.aten.mm.default(view_248, permute_124);  view_248 = permute_124 = None
    view_249: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_13, [8, 784, 768]);  mm_13 = None
    add_198: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_249, arg377_1);  view_249 = arg377_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_150: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_198);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_271: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg52_1, clone_150);  arg52_1 = clone_150 = None
    add_199: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_195, mul_271);  add_195 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_151: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_199, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_151, [2], correction = 0, keepdim = True)
    getitem_122: "f32[8, 784, 1]" = var_mean_40[0]
    getitem_123: "f32[8, 784, 1]" = var_mean_40[1];  var_mean_40 = None
    add_200: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
    rsqrt_40: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    sub_70: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_151, getitem_123);  clone_151 = getitem_123 = None
    mul_272: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_40);  sub_70 = rsqrt_40 = None
    mul_273: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_272, arg378_1);  mul_272 = arg378_1 = None
    add_201: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_273, arg379_1);  mul_273 = arg379_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_125: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_201, [0, 2, 1]);  add_201 = None
    view_250: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_125, [8, 768, 28, 28]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_30: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_250, arg380_1, arg381_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_250 = arg380_1 = arg381_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_274: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, 0.5)
    mul_275: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, 0.7071067811865476);  convolution_30 = None
    erf_28: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_275);  mul_275 = None
    add_202: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_276: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_274, add_202);  mul_274 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_38: "f32[768]" = torch.ops.prims.convert_element_type.default(arg676_1, torch.float32);  arg676_1 = None
    convert_element_type_39: "f32[768]" = torch.ops.prims.convert_element_type.default(arg677_1, torch.float32);  arg677_1 = None
    add_203: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_16: "f32[768]" = torch.ops.aten.sqrt.default(add_203);  add_203 = None
    reciprocal_16: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_277: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_135: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_136: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, -1);  unsqueeze_135 = None
    unsqueeze_137: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
    unsqueeze_138: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, -1);  unsqueeze_137 = None
    sub_71: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_276, unsqueeze_136);  mul_276 = unsqueeze_136 = None
    mul_278: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_138);  sub_71 = unsqueeze_138 = None
    unsqueeze_139: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
    unsqueeze_140: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, -1);  unsqueeze_139 = None
    mul_279: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_140);  mul_278 = unsqueeze_140 = None
    unsqueeze_141: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg383_1, -1);  arg383_1 = None
    unsqueeze_142: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, -1);  unsqueeze_141 = None
    add_204: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_142);  mul_279 = unsqueeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_31: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_204, arg384_1, arg385_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_204 = arg384_1 = arg385_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_251: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_31, [8, 768, 784]);  convolution_31 = None
    permute_126: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_280: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg54_1, permute_126);  arg54_1 = permute_126 = None
    add_205: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_199, mul_280);  add_199 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_152: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_205, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_152, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 784, 1]" = var_mean_41[0]
    getitem_125: "f32[8, 784, 1]" = var_mean_41[1];  var_mean_41 = None
    add_206: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_41: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_72: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_152, getitem_125);  clone_152 = getitem_125 = None
    mul_281: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_41);  sub_72 = rsqrt_41 = None
    mul_282: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_281, arg386_1);  mul_281 = arg386_1 = None
    add_207: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_282, arg387_1);  mul_282 = arg387_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_252: "f32[6272, 768]" = torch.ops.aten.view.default(add_207, [6272, 768]);  add_207 = None
    permute_127: "f32[768, 3072]" = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
    addmm_40: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg389_1, view_252, permute_127);  arg389_1 = view_252 = permute_127 = None
    view_253: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_40, [8, 784, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_283: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_253, 0.5)
    mul_284: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_253, 0.7071067811865476);  view_253 = None
    erf_29: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_284);  mul_284 = None
    add_208: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_285: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_283, add_208);  mul_283 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_153: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_285);  mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_254: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_153, [6272, 3072]);  clone_153 = None
    permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
    addmm_41: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg391_1, view_254, permute_128);  arg391_1 = view_254 = permute_128 = None
    view_255: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_41, [8, 784, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_154: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_255);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_286: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg55_1, clone_154);  arg55_1 = clone_154 = None
    add_209: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_205, mul_286);  add_205 = mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_155: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_209, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_155, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 784, 1]" = var_mean_42[0]
    getitem_127: "f32[8, 784, 1]" = var_mean_42[1];  var_mean_42 = None
    add_210: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_42: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_73: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_155, getitem_127);  clone_155 = getitem_127 = None
    mul_287: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_42);  sub_73 = rsqrt_42 = None
    mul_288: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_287, arg392_1);  mul_287 = arg392_1 = None
    add_211: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_288, arg393_1);  mul_288 = arg393_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_256: "f32[6272, 768]" = torch.ops.aten.view.default(add_211, [6272, 768]);  add_211 = None
    permute_129: "f32[768, 2304]" = torch.ops.aten.permute.default(arg394_1, [1, 0]);  arg394_1 = None
    addmm_42: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg395_1, view_256, permute_129);  arg395_1 = view_256 = permute_129 = None
    view_257: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_42, [8, 784, 2304]);  addmm_42 = None
    view_258: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_257, [8, 784, 3, 16, 48]);  view_257 = None
    permute_130: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_258, [2, 0, 3, 4, 1]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_130);  permute_130 = None
    getitem_128: "f32[8, 16, 48, 784]" = unbind_14[0]
    getitem_129: "f32[8, 16, 48, 784]" = unbind_14[1]
    getitem_130: "f32[8, 16, 48, 784]" = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_58: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_128, 2.0)
    sum_43: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_58, [-1], True);  pow_58 = None
    pow_59: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_43, 0.5);  sum_43 = None
    clamp_min_28: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_59, 1e-12);  pow_59 = None
    expand_84: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_28, [8, 16, 48, 784]);  clamp_min_28 = None
    div_48: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_128, expand_84);  getitem_128 = expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_60: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_129, 2.0)
    sum_44: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_60, [-1], True);  pow_60 = None
    pow_61: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_44, 0.5);  sum_44 = None
    clamp_min_29: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_61, 1e-12);  pow_61 = None
    expand_85: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_29, [8, 16, 48, 784]);  clamp_min_29 = None
    div_49: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_129, expand_85);  getitem_129 = expand_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_131: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_49, [0, 1, 3, 2]);  div_49 = None
    expand_86: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_48, [8, 16, 48, 784]);  div_48 = None
    clone_156: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    view_259: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_156, [128, 48, 784]);  clone_156 = None
    expand_87: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_131, [8, 16, 784, 48]);  permute_131 = None
    clone_157: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    view_260: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_157, [128, 784, 48]);  clone_157 = None
    bmm_28: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_259, view_260);  view_259 = view_260 = None
    view_261: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_28, [8, 16, 48, 48]);  bmm_28 = None
    mul_289: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_261, arg57_1);  view_261 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_14: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_289, [-1], True)
    sub_74: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_289, amax_14);  mul_289 = amax_14 = None
    exp_14: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
    sum_45: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_50: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_14, sum_45);  exp_14 = sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_158: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_50);  div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_88: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_158, [8, 16, 48, 48]);  clone_158 = None
    view_262: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_88, [128, 48, 48]);  expand_88 = None
    expand_89: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_130, [8, 16, 48, 784]);  getitem_130 = None
    clone_159: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_263: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_159, [128, 48, 784]);  clone_159 = None
    bmm_29: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_262, view_263);  view_262 = view_263 = None
    view_264: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_29, [8, 16, 48, 784]);  bmm_29 = None
    permute_132: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_264, [0, 3, 1, 2]);  view_264 = None
    view_265: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_132, [8, 784, 768]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
    clone_160: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_265, memory_format = torch.contiguous_format);  view_265 = None
    view_266: "f32[6272, 768]" = torch.ops.aten.view.default(clone_160, [6272, 768]);  clone_160 = None
    mm_14: "f32[6272, 768]" = torch.ops.aten.mm.default(view_266, permute_133);  view_266 = permute_133 = None
    view_267: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_14, [8, 784, 768]);  mm_14 = None
    add_212: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_267, arg397_1);  view_267 = arg397_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_161: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_212);  add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_290: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg56_1, clone_161);  arg56_1 = clone_161 = None
    add_213: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_209, mul_290);  add_209 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_162: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_213, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_162, [2], correction = 0, keepdim = True)
    getitem_131: "f32[8, 784, 1]" = var_mean_43[0]
    getitem_132: "f32[8, 784, 1]" = var_mean_43[1];  var_mean_43 = None
    add_214: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_131, 1e-06);  getitem_131 = None
    rsqrt_43: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
    sub_75: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_162, getitem_132);  clone_162 = getitem_132 = None
    mul_291: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_43);  sub_75 = rsqrt_43 = None
    mul_292: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_291, arg398_1);  mul_291 = arg398_1 = None
    add_215: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_292, arg399_1);  mul_292 = arg399_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_134: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_215, [0, 2, 1]);  add_215 = None
    view_268: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_134, [8, 768, 28, 28]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_32: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_268, arg400_1, arg401_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_268 = arg400_1 = arg401_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_293: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, 0.5)
    mul_294: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, 0.7071067811865476);  convolution_32 = None
    erf_30: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
    add_216: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_295: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_293, add_216);  mul_293 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_40: "f32[768]" = torch.ops.prims.convert_element_type.default(arg679_1, torch.float32);  arg679_1 = None
    convert_element_type_41: "f32[768]" = torch.ops.prims.convert_element_type.default(arg680_1, torch.float32);  arg680_1 = None
    add_217: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_17: "f32[768]" = torch.ops.aten.sqrt.default(add_217);  add_217 = None
    reciprocal_17: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_296: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_143: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_144: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, -1);  unsqueeze_143 = None
    unsqueeze_145: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_296, -1);  mul_296 = None
    unsqueeze_146: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, -1);  unsqueeze_145 = None
    sub_76: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_295, unsqueeze_144);  mul_295 = unsqueeze_144 = None
    mul_297: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_146);  sub_76 = unsqueeze_146 = None
    unsqueeze_147: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
    unsqueeze_148: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, -1);  unsqueeze_147 = None
    mul_298: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_297, unsqueeze_148);  mul_297 = unsqueeze_148 = None
    unsqueeze_149: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
    unsqueeze_150: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, -1);  unsqueeze_149 = None
    add_218: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_150);  mul_298 = unsqueeze_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_33: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_218, arg404_1, arg405_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_218 = arg404_1 = arg405_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_269: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_33, [8, 768, 784]);  convolution_33 = None
    permute_135: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_299: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg58_1, permute_135);  arg58_1 = permute_135 = None
    add_219: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_213, mul_299);  add_213 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_163: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_219, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_163, [2], correction = 0, keepdim = True)
    getitem_133: "f32[8, 784, 1]" = var_mean_44[0]
    getitem_134: "f32[8, 784, 1]" = var_mean_44[1];  var_mean_44 = None
    add_220: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_133, 1e-06);  getitem_133 = None
    rsqrt_44: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_77: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_163, getitem_134);  clone_163 = getitem_134 = None
    mul_300: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_44);  sub_77 = rsqrt_44 = None
    mul_301: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_300, arg406_1);  mul_300 = arg406_1 = None
    add_221: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_301, arg407_1);  mul_301 = arg407_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_270: "f32[6272, 768]" = torch.ops.aten.view.default(add_221, [6272, 768]);  add_221 = None
    permute_136: "f32[768, 3072]" = torch.ops.aten.permute.default(arg408_1, [1, 0]);  arg408_1 = None
    addmm_43: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg409_1, view_270, permute_136);  arg409_1 = view_270 = permute_136 = None
    view_271: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_43, [8, 784, 3072]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_302: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_271, 0.5)
    mul_303: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_271, 0.7071067811865476);  view_271 = None
    erf_31: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
    add_222: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_304: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_302, add_222);  mul_302 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_164: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_304);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_272: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_164, [6272, 3072]);  clone_164 = None
    permute_137: "f32[3072, 768]" = torch.ops.aten.permute.default(arg410_1, [1, 0]);  arg410_1 = None
    addmm_44: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg411_1, view_272, permute_137);  arg411_1 = view_272 = permute_137 = None
    view_273: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_44, [8, 784, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_165: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_273);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_305: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg59_1, clone_165);  arg59_1 = clone_165 = None
    add_223: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_219, mul_305);  add_219 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_166: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_223, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_166, [2], correction = 0, keepdim = True)
    getitem_135: "f32[8, 784, 1]" = var_mean_45[0]
    getitem_136: "f32[8, 784, 1]" = var_mean_45[1];  var_mean_45 = None
    add_224: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_135, 1e-06);  getitem_135 = None
    rsqrt_45: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_78: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_166, getitem_136);  clone_166 = getitem_136 = None
    mul_306: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_45);  sub_78 = rsqrt_45 = None
    mul_307: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_306, arg412_1);  mul_306 = arg412_1 = None
    add_225: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_307, arg413_1);  mul_307 = arg413_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_274: "f32[6272, 768]" = torch.ops.aten.view.default(add_225, [6272, 768]);  add_225 = None
    permute_138: "f32[768, 2304]" = torch.ops.aten.permute.default(arg414_1, [1, 0]);  arg414_1 = None
    addmm_45: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg415_1, view_274, permute_138);  arg415_1 = view_274 = permute_138 = None
    view_275: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_45, [8, 784, 2304]);  addmm_45 = None
    view_276: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_275, [8, 784, 3, 16, 48]);  view_275 = None
    permute_139: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_276, [2, 0, 3, 4, 1]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_139);  permute_139 = None
    getitem_137: "f32[8, 16, 48, 784]" = unbind_15[0]
    getitem_138: "f32[8, 16, 48, 784]" = unbind_15[1]
    getitem_139: "f32[8, 16, 48, 784]" = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_62: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_137, 2.0)
    sum_46: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_62, [-1], True);  pow_62 = None
    pow_63: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_46, 0.5);  sum_46 = None
    clamp_min_30: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_63, 1e-12);  pow_63 = None
    expand_90: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_30, [8, 16, 48, 784]);  clamp_min_30 = None
    div_51: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_137, expand_90);  getitem_137 = expand_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_64: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_138, 2.0)
    sum_47: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_64, [-1], True);  pow_64 = None
    pow_65: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_47, 0.5);  sum_47 = None
    clamp_min_31: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_65, 1e-12);  pow_65 = None
    expand_91: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_31, [8, 16, 48, 784]);  clamp_min_31 = None
    div_52: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_138, expand_91);  getitem_138 = expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_140: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_52, [0, 1, 3, 2]);  div_52 = None
    expand_92: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_51, [8, 16, 48, 784]);  div_51 = None
    clone_167: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_277: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_167, [128, 48, 784]);  clone_167 = None
    expand_93: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_140, [8, 16, 784, 48]);  permute_140 = None
    clone_168: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_278: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_168, [128, 784, 48]);  clone_168 = None
    bmm_30: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_277, view_278);  view_277 = view_278 = None
    view_279: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_30, [8, 16, 48, 48]);  bmm_30 = None
    mul_308: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_279, arg61_1);  view_279 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_15: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_308, [-1], True)
    sub_79: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_308, amax_15);  mul_308 = amax_15 = None
    exp_15: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
    sum_48: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_53: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_15, sum_48);  exp_15 = sum_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_169: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_53);  div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_94: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_169, [8, 16, 48, 48]);  clone_169 = None
    view_280: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_94, [128, 48, 48]);  expand_94 = None
    expand_95: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_139, [8, 16, 48, 784]);  getitem_139 = None
    clone_170: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    view_281: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_170, [128, 48, 784]);  clone_170 = None
    bmm_31: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_280, view_281);  view_280 = view_281 = None
    view_282: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_31, [8, 16, 48, 784]);  bmm_31 = None
    permute_141: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_282, [0, 3, 1, 2]);  view_282 = None
    view_283: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_141, [8, 784, 768]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
    clone_171: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_283, memory_format = torch.contiguous_format);  view_283 = None
    view_284: "f32[6272, 768]" = torch.ops.aten.view.default(clone_171, [6272, 768]);  clone_171 = None
    mm_15: "f32[6272, 768]" = torch.ops.aten.mm.default(view_284, permute_142);  view_284 = permute_142 = None
    view_285: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_15, [8, 784, 768]);  mm_15 = None
    add_226: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_285, arg417_1);  view_285 = arg417_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_172: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_309: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg60_1, clone_172);  arg60_1 = clone_172 = None
    add_227: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_223, mul_309);  add_223 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_173: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_227, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_173, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 784, 1]" = var_mean_46[0]
    getitem_141: "f32[8, 784, 1]" = var_mean_46[1];  var_mean_46 = None
    add_228: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_46: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_80: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_173, getitem_141);  clone_173 = getitem_141 = None
    mul_310: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_46);  sub_80 = rsqrt_46 = None
    mul_311: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_310, arg418_1);  mul_310 = arg418_1 = None
    add_229: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_311, arg419_1);  mul_311 = arg419_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_143: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_229, [0, 2, 1]);  add_229 = None
    view_286: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_143, [8, 768, 28, 28]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_34: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_286, arg420_1, arg421_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_286 = arg420_1 = arg421_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_312: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, 0.5)
    mul_313: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476);  convolution_34 = None
    erf_32: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_313);  mul_313 = None
    add_230: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_314: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_312, add_230);  mul_312 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_42: "f32[768]" = torch.ops.prims.convert_element_type.default(arg682_1, torch.float32);  arg682_1 = None
    convert_element_type_43: "f32[768]" = torch.ops.prims.convert_element_type.default(arg683_1, torch.float32);  arg683_1 = None
    add_231: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_18: "f32[768]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
    reciprocal_18: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_315: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_151: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_152: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, -1);  unsqueeze_151 = None
    unsqueeze_153: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
    unsqueeze_154: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, -1);  unsqueeze_153 = None
    sub_81: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_314, unsqueeze_152);  mul_314 = unsqueeze_152 = None
    mul_316: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_154);  sub_81 = unsqueeze_154 = None
    unsqueeze_155: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
    unsqueeze_156: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, -1);  unsqueeze_155 = None
    mul_317: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_156);  mul_316 = unsqueeze_156 = None
    unsqueeze_157: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
    unsqueeze_158: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, -1);  unsqueeze_157 = None
    add_232: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_158);  mul_317 = unsqueeze_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_35: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_232, arg424_1, arg425_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_232 = arg424_1 = arg425_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_287: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_35, [8, 768, 784]);  convolution_35 = None
    permute_144: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_318: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg62_1, permute_144);  arg62_1 = permute_144 = None
    add_233: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_227, mul_318);  add_227 = mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_174: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_233, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 784, 1]" = var_mean_47[0]
    getitem_143: "f32[8, 784, 1]" = var_mean_47[1];  var_mean_47 = None
    add_234: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
    rsqrt_47: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    sub_82: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_174, getitem_143);  clone_174 = getitem_143 = None
    mul_319: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_47);  sub_82 = rsqrt_47 = None
    mul_320: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_319, arg426_1);  mul_319 = arg426_1 = None
    add_235: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_320, arg427_1);  mul_320 = arg427_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_288: "f32[6272, 768]" = torch.ops.aten.view.default(add_235, [6272, 768]);  add_235 = None
    permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(arg428_1, [1, 0]);  arg428_1 = None
    addmm_46: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg429_1, view_288, permute_145);  arg429_1 = view_288 = permute_145 = None
    view_289: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 784, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_321: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_289, 0.5)
    mul_322: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_289, 0.7071067811865476);  view_289 = None
    erf_33: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_322);  mul_322 = None
    add_236: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_323: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_321, add_236);  mul_321 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_175: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_323);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_290: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_175, [6272, 3072]);  clone_175 = None
    permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(arg430_1, [1, 0]);  arg430_1 = None
    addmm_47: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg431_1, view_290, permute_146);  arg431_1 = view_290 = permute_146 = None
    view_291: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_47, [8, 784, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_176: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_291);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_324: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg63_1, clone_176);  arg63_1 = clone_176 = None
    add_237: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_233, mul_324);  add_233 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_177: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_237, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 784, 1]" = var_mean_48[0]
    getitem_145: "f32[8, 784, 1]" = var_mean_48[1];  var_mean_48 = None
    add_238: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_48: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
    sub_83: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_177, getitem_145);  clone_177 = getitem_145 = None
    mul_325: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_48);  sub_83 = rsqrt_48 = None
    mul_326: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_325, arg432_1);  mul_325 = arg432_1 = None
    add_239: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_326, arg433_1);  mul_326 = arg433_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_292: "f32[6272, 768]" = torch.ops.aten.view.default(add_239, [6272, 768]);  add_239 = None
    permute_147: "f32[768, 2304]" = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
    addmm_48: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg435_1, view_292, permute_147);  arg435_1 = view_292 = permute_147 = None
    view_293: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_48, [8, 784, 2304]);  addmm_48 = None
    view_294: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_293, [8, 784, 3, 16, 48]);  view_293 = None
    permute_148: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_294, [2, 0, 3, 4, 1]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_148);  permute_148 = None
    getitem_146: "f32[8, 16, 48, 784]" = unbind_16[0]
    getitem_147: "f32[8, 16, 48, 784]" = unbind_16[1]
    getitem_148: "f32[8, 16, 48, 784]" = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_66: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_146, 2.0)
    sum_49: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_66, [-1], True);  pow_66 = None
    pow_67: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_49, 0.5);  sum_49 = None
    clamp_min_32: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_67, 1e-12);  pow_67 = None
    expand_96: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_32, [8, 16, 48, 784]);  clamp_min_32 = None
    div_54: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_146, expand_96);  getitem_146 = expand_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_68: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_147, 2.0)
    sum_50: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_68, [-1], True);  pow_68 = None
    pow_69: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_50, 0.5);  sum_50 = None
    clamp_min_33: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_69, 1e-12);  pow_69 = None
    expand_97: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_33, [8, 16, 48, 784]);  clamp_min_33 = None
    div_55: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_147, expand_97);  getitem_147 = expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_149: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_55, [0, 1, 3, 2]);  div_55 = None
    expand_98: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_54, [8, 16, 48, 784]);  div_54 = None
    clone_178: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
    view_295: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_178, [128, 48, 784]);  clone_178 = None
    expand_99: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_149, [8, 16, 784, 48]);  permute_149 = None
    clone_179: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
    view_296: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_179, [128, 784, 48]);  clone_179 = None
    bmm_32: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_295, view_296);  view_295 = view_296 = None
    view_297: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_32, [8, 16, 48, 48]);  bmm_32 = None
    mul_327: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_297, arg65_1);  view_297 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_16: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_327, [-1], True)
    sub_84: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_327, amax_16);  mul_327 = amax_16 = None
    exp_16: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
    sum_51: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_56: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_16, sum_51);  exp_16 = sum_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_180: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_56);  div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_100: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_180, [8, 16, 48, 48]);  clone_180 = None
    view_298: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_100, [128, 48, 48]);  expand_100 = None
    expand_101: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_148, [8, 16, 48, 784]);  getitem_148 = None
    clone_181: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
    view_299: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_181, [128, 48, 784]);  clone_181 = None
    bmm_33: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_298, view_299);  view_298 = view_299 = None
    view_300: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_33, [8, 16, 48, 784]);  bmm_33 = None
    permute_150: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_300, [0, 3, 1, 2]);  view_300 = None
    view_301: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_150, [8, 784, 768]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(arg436_1, [1, 0]);  arg436_1 = None
    clone_182: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_301, memory_format = torch.contiguous_format);  view_301 = None
    view_302: "f32[6272, 768]" = torch.ops.aten.view.default(clone_182, [6272, 768]);  clone_182 = None
    mm_16: "f32[6272, 768]" = torch.ops.aten.mm.default(view_302, permute_151);  view_302 = permute_151 = None
    view_303: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_16, [8, 784, 768]);  mm_16 = None
    add_240: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_303, arg437_1);  view_303 = arg437_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_183: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_240);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_328: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg64_1, clone_183);  arg64_1 = clone_183 = None
    add_241: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_237, mul_328);  add_237 = mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_184: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_241, memory_format = torch.contiguous_format)
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_184, [2], correction = 0, keepdim = True)
    getitem_149: "f32[8, 784, 1]" = var_mean_49[0]
    getitem_150: "f32[8, 784, 1]" = var_mean_49[1];  var_mean_49 = None
    add_242: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_149, 1e-06);  getitem_149 = None
    rsqrt_49: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_85: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_184, getitem_150);  clone_184 = getitem_150 = None
    mul_329: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_49);  sub_85 = rsqrt_49 = None
    mul_330: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_329, arg438_1);  mul_329 = arg438_1 = None
    add_243: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_330, arg439_1);  mul_330 = arg439_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_152: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_243, [0, 2, 1]);  add_243 = None
    view_304: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_152, [8, 768, 28, 28]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_36: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_304, arg440_1, arg441_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_304 = arg440_1 = arg441_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_331: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, 0.5)
    mul_332: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, 0.7071067811865476);  convolution_36 = None
    erf_34: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_332);  mul_332 = None
    add_244: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_333: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_331, add_244);  mul_331 = add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_44: "f32[768]" = torch.ops.prims.convert_element_type.default(arg685_1, torch.float32);  arg685_1 = None
    convert_element_type_45: "f32[768]" = torch.ops.prims.convert_element_type.default(arg686_1, torch.float32);  arg686_1 = None
    add_245: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_19: "f32[768]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
    reciprocal_19: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_334: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_159: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_160: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, -1);  unsqueeze_159 = None
    unsqueeze_161: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_334, -1);  mul_334 = None
    unsqueeze_162: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, -1);  unsqueeze_161 = None
    sub_86: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_333, unsqueeze_160);  mul_333 = unsqueeze_160 = None
    mul_335: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_162);  sub_86 = unsqueeze_162 = None
    unsqueeze_163: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
    unsqueeze_164: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, -1);  unsqueeze_163 = None
    mul_336: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_164);  mul_335 = unsqueeze_164 = None
    unsqueeze_165: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
    unsqueeze_166: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, -1);  unsqueeze_165 = None
    add_246: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_336, unsqueeze_166);  mul_336 = unsqueeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_37: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_246, arg444_1, arg445_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_246 = arg444_1 = arg445_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_305: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_37, [8, 768, 784]);  convolution_37 = None
    permute_153: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_337: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg66_1, permute_153);  arg66_1 = permute_153 = None
    add_247: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_241, mul_337);  add_241 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_185: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_247, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_185, [2], correction = 0, keepdim = True)
    getitem_151: "f32[8, 784, 1]" = var_mean_50[0]
    getitem_152: "f32[8, 784, 1]" = var_mean_50[1];  var_mean_50 = None
    add_248: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_151, 1e-06);  getitem_151 = None
    rsqrt_50: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
    sub_87: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_185, getitem_152);  clone_185 = getitem_152 = None
    mul_338: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_50);  sub_87 = rsqrt_50 = None
    mul_339: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_338, arg446_1);  mul_338 = arg446_1 = None
    add_249: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_339, arg447_1);  mul_339 = arg447_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_306: "f32[6272, 768]" = torch.ops.aten.view.default(add_249, [6272, 768]);  add_249 = None
    permute_154: "f32[768, 3072]" = torch.ops.aten.permute.default(arg448_1, [1, 0]);  arg448_1 = None
    addmm_49: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg449_1, view_306, permute_154);  arg449_1 = view_306 = permute_154 = None
    view_307: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_49, [8, 784, 3072]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_340: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    mul_341: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_307, 0.7071067811865476);  view_307 = None
    erf_35: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_341);  mul_341 = None
    add_250: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_342: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_340, add_250);  mul_340 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_186: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_342);  mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_308: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_186, [6272, 3072]);  clone_186 = None
    permute_155: "f32[3072, 768]" = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
    addmm_50: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg451_1, view_308, permute_155);  arg451_1 = view_308 = permute_155 = None
    view_309: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_50, [8, 784, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_187: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_343: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg67_1, clone_187);  arg67_1 = clone_187 = None
    add_251: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_247, mul_343);  add_247 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_188: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_251, memory_format = torch.contiguous_format)
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_188, [2], correction = 0, keepdim = True)
    getitem_153: "f32[8, 784, 1]" = var_mean_51[0]
    getitem_154: "f32[8, 784, 1]" = var_mean_51[1];  var_mean_51 = None
    add_252: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-06);  getitem_153 = None
    rsqrt_51: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_88: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_188, getitem_154);  clone_188 = getitem_154 = None
    mul_344: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_51);  sub_88 = rsqrt_51 = None
    mul_345: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_344, arg452_1);  mul_344 = arg452_1 = None
    add_253: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_345, arg453_1);  mul_345 = arg453_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_310: "f32[6272, 768]" = torch.ops.aten.view.default(add_253, [6272, 768]);  add_253 = None
    permute_156: "f32[768, 2304]" = torch.ops.aten.permute.default(arg454_1, [1, 0]);  arg454_1 = None
    addmm_51: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg455_1, view_310, permute_156);  arg455_1 = view_310 = permute_156 = None
    view_311: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_51, [8, 784, 2304]);  addmm_51 = None
    view_312: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_311, [8, 784, 3, 16, 48]);  view_311 = None
    permute_157: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_312, [2, 0, 3, 4, 1]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
    getitem_155: "f32[8, 16, 48, 784]" = unbind_17[0]
    getitem_156: "f32[8, 16, 48, 784]" = unbind_17[1]
    getitem_157: "f32[8, 16, 48, 784]" = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_70: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_155, 2.0)
    sum_52: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_70, [-1], True);  pow_70 = None
    pow_71: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_52, 0.5);  sum_52 = None
    clamp_min_34: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_71, 1e-12);  pow_71 = None
    expand_102: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_34, [8, 16, 48, 784]);  clamp_min_34 = None
    div_57: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_155, expand_102);  getitem_155 = expand_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_72: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_156, 2.0)
    sum_53: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_72, [-1], True);  pow_72 = None
    pow_73: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_53, 0.5);  sum_53 = None
    clamp_min_35: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_73, 1e-12);  pow_73 = None
    expand_103: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_35, [8, 16, 48, 784]);  clamp_min_35 = None
    div_58: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_156, expand_103);  getitem_156 = expand_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_158: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_58, [0, 1, 3, 2]);  div_58 = None
    expand_104: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_57, [8, 16, 48, 784]);  div_57 = None
    clone_189: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    view_313: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_189, [128, 48, 784]);  clone_189 = None
    expand_105: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_158, [8, 16, 784, 48]);  permute_158 = None
    clone_190: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    view_314: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_190, [128, 784, 48]);  clone_190 = None
    bmm_34: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_313, view_314);  view_313 = view_314 = None
    view_315: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_34, [8, 16, 48, 48]);  bmm_34 = None
    mul_346: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_315, arg69_1);  view_315 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_17: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_346, [-1], True)
    sub_89: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_346, amax_17);  mul_346 = amax_17 = None
    exp_17: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
    sum_54: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_59: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_17, sum_54);  exp_17 = sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_191: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_59);  div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_106: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_191, [8, 16, 48, 48]);  clone_191 = None
    view_316: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_106, [128, 48, 48]);  expand_106 = None
    expand_107: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_157, [8, 16, 48, 784]);  getitem_157 = None
    clone_192: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
    view_317: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_192, [128, 48, 784]);  clone_192 = None
    bmm_35: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_316, view_317);  view_316 = view_317 = None
    view_318: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_35, [8, 16, 48, 784]);  bmm_35 = None
    permute_159: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_318, [0, 3, 1, 2]);  view_318 = None
    view_319: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_159, [8, 784, 768]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(arg456_1, [1, 0]);  arg456_1 = None
    clone_193: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_319, memory_format = torch.contiguous_format);  view_319 = None
    view_320: "f32[6272, 768]" = torch.ops.aten.view.default(clone_193, [6272, 768]);  clone_193 = None
    mm_17: "f32[6272, 768]" = torch.ops.aten.mm.default(view_320, permute_160);  view_320 = permute_160 = None
    view_321: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_17, [8, 784, 768]);  mm_17 = None
    add_254: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_321, arg457_1);  view_321 = arg457_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_194: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_254);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_347: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg68_1, clone_194);  arg68_1 = clone_194 = None
    add_255: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_251, mul_347);  add_251 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_195: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_255, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_195, [2], correction = 0, keepdim = True)
    getitem_158: "f32[8, 784, 1]" = var_mean_52[0]
    getitem_159: "f32[8, 784, 1]" = var_mean_52[1];  var_mean_52 = None
    add_256: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
    rsqrt_52: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_90: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_195, getitem_159);  clone_195 = getitem_159 = None
    mul_348: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_52);  sub_90 = rsqrt_52 = None
    mul_349: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_348, arg458_1);  mul_348 = arg458_1 = None
    add_257: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_349, arg459_1);  mul_349 = arg459_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_161: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_257, [0, 2, 1]);  add_257 = None
    view_322: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_161, [8, 768, 28, 28]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_38: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_322, arg460_1, arg461_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_322 = arg460_1 = arg461_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_350: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, 0.5)
    mul_351: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476);  convolution_38 = None
    erf_36: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_351);  mul_351 = None
    add_258: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_352: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_350, add_258);  mul_350 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_46: "f32[768]" = torch.ops.prims.convert_element_type.default(arg688_1, torch.float32);  arg688_1 = None
    convert_element_type_47: "f32[768]" = torch.ops.prims.convert_element_type.default(arg689_1, torch.float32);  arg689_1 = None
    add_259: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_20: "f32[768]" = torch.ops.aten.sqrt.default(add_259);  add_259 = None
    reciprocal_20: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_353: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_167: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_168: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, -1);  unsqueeze_167 = None
    unsqueeze_169: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_353, -1);  mul_353 = None
    unsqueeze_170: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, -1);  unsqueeze_169 = None
    sub_91: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_352, unsqueeze_168);  mul_352 = unsqueeze_168 = None
    mul_354: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_170);  sub_91 = unsqueeze_170 = None
    unsqueeze_171: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg462_1, -1);  arg462_1 = None
    unsqueeze_172: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, -1);  unsqueeze_171 = None
    mul_355: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_354, unsqueeze_172);  mul_354 = unsqueeze_172 = None
    unsqueeze_173: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg463_1, -1);  arg463_1 = None
    unsqueeze_174: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, -1);  unsqueeze_173 = None
    add_260: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_355, unsqueeze_174);  mul_355 = unsqueeze_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_39: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_260, arg464_1, arg465_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_260 = arg464_1 = arg465_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_323: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_39, [8, 768, 784]);  convolution_39 = None
    permute_162: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_356: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg70_1, permute_162);  arg70_1 = permute_162 = None
    add_261: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_255, mul_356);  add_255 = mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_196: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_261, memory_format = torch.contiguous_format)
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_196, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 784, 1]" = var_mean_53[0]
    getitem_161: "f32[8, 784, 1]" = var_mean_53[1];  var_mean_53 = None
    add_262: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_53: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    sub_92: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_196, getitem_161);  clone_196 = getitem_161 = None
    mul_357: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_53);  sub_92 = rsqrt_53 = None
    mul_358: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_357, arg466_1);  mul_357 = arg466_1 = None
    add_263: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_358, arg467_1);  mul_358 = arg467_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_324: "f32[6272, 768]" = torch.ops.aten.view.default(add_263, [6272, 768]);  add_263 = None
    permute_163: "f32[768, 3072]" = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
    addmm_52: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg469_1, view_324, permute_163);  arg469_1 = view_324 = permute_163 = None
    view_325: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_52, [8, 784, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_359: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_325, 0.5)
    mul_360: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_325, 0.7071067811865476);  view_325 = None
    erf_37: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_360);  mul_360 = None
    add_264: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_361: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_359, add_264);  mul_359 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_197: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_361);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_326: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_197, [6272, 3072]);  clone_197 = None
    permute_164: "f32[3072, 768]" = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
    addmm_53: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg471_1, view_326, permute_164);  arg471_1 = view_326 = permute_164 = None
    view_327: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_53, [8, 784, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_198: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_327);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_362: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg71_1, clone_198);  arg71_1 = clone_198 = None
    add_265: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_261, mul_362);  add_261 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_199: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_265, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_199, [2], correction = 0, keepdim = True)
    getitem_162: "f32[8, 784, 1]" = var_mean_54[0]
    getitem_163: "f32[8, 784, 1]" = var_mean_54[1];  var_mean_54 = None
    add_266: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_54: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
    sub_93: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_199, getitem_163);  clone_199 = getitem_163 = None
    mul_363: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_54);  sub_93 = rsqrt_54 = None
    mul_364: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_363, arg472_1);  mul_363 = arg472_1 = None
    add_267: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_364, arg473_1);  mul_364 = arg473_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_328: "f32[6272, 768]" = torch.ops.aten.view.default(add_267, [6272, 768]);  add_267 = None
    permute_165: "f32[768, 2304]" = torch.ops.aten.permute.default(arg474_1, [1, 0]);  arg474_1 = None
    addmm_54: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg475_1, view_328, permute_165);  arg475_1 = view_328 = permute_165 = None
    view_329: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_54, [8, 784, 2304]);  addmm_54 = None
    view_330: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_329, [8, 784, 3, 16, 48]);  view_329 = None
    permute_166: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_330, [2, 0, 3, 4, 1]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_166);  permute_166 = None
    getitem_164: "f32[8, 16, 48, 784]" = unbind_18[0]
    getitem_165: "f32[8, 16, 48, 784]" = unbind_18[1]
    getitem_166: "f32[8, 16, 48, 784]" = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_74: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_164, 2.0)
    sum_55: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_74, [-1], True);  pow_74 = None
    pow_75: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_55, 0.5);  sum_55 = None
    clamp_min_36: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_75, 1e-12);  pow_75 = None
    expand_108: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_36, [8, 16, 48, 784]);  clamp_min_36 = None
    div_60: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_164, expand_108);  getitem_164 = expand_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_76: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_165, 2.0)
    sum_56: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_76, [-1], True);  pow_76 = None
    pow_77: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_56, 0.5);  sum_56 = None
    clamp_min_37: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_77, 1e-12);  pow_77 = None
    expand_109: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_37, [8, 16, 48, 784]);  clamp_min_37 = None
    div_61: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_165, expand_109);  getitem_165 = expand_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_167: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_61, [0, 1, 3, 2]);  div_61 = None
    expand_110: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_60, [8, 16, 48, 784]);  div_60 = None
    clone_200: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
    view_331: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_200, [128, 48, 784]);  clone_200 = None
    expand_111: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_167, [8, 16, 784, 48]);  permute_167 = None
    clone_201: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
    view_332: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_201, [128, 784, 48]);  clone_201 = None
    bmm_36: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
    view_333: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_36, [8, 16, 48, 48]);  bmm_36 = None
    mul_365: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_333, arg73_1);  view_333 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_18: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_365, [-1], True)
    sub_94: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_365, amax_18);  mul_365 = amax_18 = None
    exp_18: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_94);  sub_94 = None
    sum_57: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_62: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_18, sum_57);  exp_18 = sum_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_202: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_62);  div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_112: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_202, [8, 16, 48, 48]);  clone_202 = None
    view_334: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_112, [128, 48, 48]);  expand_112 = None
    expand_113: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_166, [8, 16, 48, 784]);  getitem_166 = None
    clone_203: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    view_335: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_203, [128, 48, 784]);  clone_203 = None
    bmm_37: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_334, view_335);  view_334 = view_335 = None
    view_336: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_37, [8, 16, 48, 784]);  bmm_37 = None
    permute_168: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_336, [0, 3, 1, 2]);  view_336 = None
    view_337: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_168, [8, 784, 768]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(arg476_1, [1, 0]);  arg476_1 = None
    clone_204: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_337, memory_format = torch.contiguous_format);  view_337 = None
    view_338: "f32[6272, 768]" = torch.ops.aten.view.default(clone_204, [6272, 768]);  clone_204 = None
    mm_18: "f32[6272, 768]" = torch.ops.aten.mm.default(view_338, permute_169);  view_338 = permute_169 = None
    view_339: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_18, [8, 784, 768]);  mm_18 = None
    add_268: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_339, arg477_1);  view_339 = arg477_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_205: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_268);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_366: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg72_1, clone_205);  arg72_1 = clone_205 = None
    add_269: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_265, mul_366);  add_265 = mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_206: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_269, memory_format = torch.contiguous_format)
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_206, [2], correction = 0, keepdim = True)
    getitem_167: "f32[8, 784, 1]" = var_mean_55[0]
    getitem_168: "f32[8, 784, 1]" = var_mean_55[1];  var_mean_55 = None
    add_270: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
    rsqrt_55: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_270);  add_270 = None
    sub_95: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_206, getitem_168);  clone_206 = getitem_168 = None
    mul_367: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_55);  sub_95 = rsqrt_55 = None
    mul_368: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_367, arg478_1);  mul_367 = arg478_1 = None
    add_271: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_368, arg479_1);  mul_368 = arg479_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_170: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_271, [0, 2, 1]);  add_271 = None
    view_340: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_170, [8, 768, 28, 28]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_40: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_340, arg480_1, arg481_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_340 = arg480_1 = arg481_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_369: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, 0.5)
    mul_370: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, 0.7071067811865476);  convolution_40 = None
    erf_38: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_370);  mul_370 = None
    add_272: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_371: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_369, add_272);  mul_369 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_48: "f32[768]" = torch.ops.prims.convert_element_type.default(arg691_1, torch.float32);  arg691_1 = None
    convert_element_type_49: "f32[768]" = torch.ops.prims.convert_element_type.default(arg692_1, torch.float32);  arg692_1 = None
    add_273: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_21: "f32[768]" = torch.ops.aten.sqrt.default(add_273);  add_273 = None
    reciprocal_21: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_372: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_175: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_176: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, -1);  unsqueeze_175 = None
    unsqueeze_177: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
    unsqueeze_178: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, -1);  unsqueeze_177 = None
    sub_96: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_371, unsqueeze_176);  mul_371 = unsqueeze_176 = None
    mul_373: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_178);  sub_96 = unsqueeze_178 = None
    unsqueeze_179: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
    unsqueeze_180: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, -1);  unsqueeze_179 = None
    mul_374: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_180);  mul_373 = unsqueeze_180 = None
    unsqueeze_181: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg483_1, -1);  arg483_1 = None
    unsqueeze_182: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, -1);  unsqueeze_181 = None
    add_274: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_182);  mul_374 = unsqueeze_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_41: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_274, arg484_1, arg485_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_274 = arg484_1 = arg485_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_341: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_41, [8, 768, 784]);  convolution_41 = None
    permute_171: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_375: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg74_1, permute_171);  arg74_1 = permute_171 = None
    add_275: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_269, mul_375);  add_269 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_207: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_275, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_207, [2], correction = 0, keepdim = True)
    getitem_169: "f32[8, 784, 1]" = var_mean_56[0]
    getitem_170: "f32[8, 784, 1]" = var_mean_56[1];  var_mean_56 = None
    add_276: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-06);  getitem_169 = None
    rsqrt_56: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    sub_97: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_207, getitem_170);  clone_207 = getitem_170 = None
    mul_376: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_56);  sub_97 = rsqrt_56 = None
    mul_377: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_376, arg486_1);  mul_376 = arg486_1 = None
    add_277: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_377, arg487_1);  mul_377 = arg487_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_342: "f32[6272, 768]" = torch.ops.aten.view.default(add_277, [6272, 768]);  add_277 = None
    permute_172: "f32[768, 3072]" = torch.ops.aten.permute.default(arg488_1, [1, 0]);  arg488_1 = None
    addmm_55: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg489_1, view_342, permute_172);  arg489_1 = view_342 = permute_172 = None
    view_343: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_55, [8, 784, 3072]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_378: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    mul_379: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476);  view_343 = None
    erf_39: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_379);  mul_379 = None
    add_278: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_380: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_378, add_278);  mul_378 = add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_208: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_380);  mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_344: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_208, [6272, 3072]);  clone_208 = None
    permute_173: "f32[3072, 768]" = torch.ops.aten.permute.default(arg490_1, [1, 0]);  arg490_1 = None
    addmm_56: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg491_1, view_344, permute_173);  arg491_1 = view_344 = permute_173 = None
    view_345: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_56, [8, 784, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_209: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_345);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_381: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg75_1, clone_209);  arg75_1 = clone_209 = None
    add_279: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_275, mul_381);  add_275 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_210: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format)
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_210, [2], correction = 0, keepdim = True)
    getitem_171: "f32[8, 784, 1]" = var_mean_57[0]
    getitem_172: "f32[8, 784, 1]" = var_mean_57[1];  var_mean_57 = None
    add_280: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_171, 1e-06);  getitem_171 = None
    rsqrt_57: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
    sub_98: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_210, getitem_172);  clone_210 = getitem_172 = None
    mul_382: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_57);  sub_98 = rsqrt_57 = None
    mul_383: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_382, arg492_1);  mul_382 = arg492_1 = None
    add_281: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_383, arg493_1);  mul_383 = arg493_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_346: "f32[6272, 768]" = torch.ops.aten.view.default(add_281, [6272, 768]);  add_281 = None
    permute_174: "f32[768, 2304]" = torch.ops.aten.permute.default(arg494_1, [1, 0]);  arg494_1 = None
    addmm_57: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg495_1, view_346, permute_174);  arg495_1 = view_346 = permute_174 = None
    view_347: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_57, [8, 784, 2304]);  addmm_57 = None
    view_348: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_347, [8, 784, 3, 16, 48]);  view_347 = None
    permute_175: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_348, [2, 0, 3, 4, 1]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_175);  permute_175 = None
    getitem_173: "f32[8, 16, 48, 784]" = unbind_19[0]
    getitem_174: "f32[8, 16, 48, 784]" = unbind_19[1]
    getitem_175: "f32[8, 16, 48, 784]" = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_78: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_173, 2.0)
    sum_58: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_78, [-1], True);  pow_78 = None
    pow_79: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_58, 0.5);  sum_58 = None
    clamp_min_38: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_79, 1e-12);  pow_79 = None
    expand_114: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_38, [8, 16, 48, 784]);  clamp_min_38 = None
    div_63: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_173, expand_114);  getitem_173 = expand_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_80: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_174, 2.0)
    sum_59: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_80, [-1], True);  pow_80 = None
    pow_81: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_59, 0.5);  sum_59 = None
    clamp_min_39: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_81, 1e-12);  pow_81 = None
    expand_115: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_39, [8, 16, 48, 784]);  clamp_min_39 = None
    div_64: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_174, expand_115);  getitem_174 = expand_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_176: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_64, [0, 1, 3, 2]);  div_64 = None
    expand_116: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_63, [8, 16, 48, 784]);  div_63 = None
    clone_211: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
    view_349: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_211, [128, 48, 784]);  clone_211 = None
    expand_117: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_176, [8, 16, 784, 48]);  permute_176 = None
    clone_212: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
    view_350: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_212, [128, 784, 48]);  clone_212 = None
    bmm_38: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_349, view_350);  view_349 = view_350 = None
    view_351: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_38, [8, 16, 48, 48]);  bmm_38 = None
    mul_384: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_351, arg77_1);  view_351 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_19: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_384, [-1], True)
    sub_99: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_384, amax_19);  mul_384 = amax_19 = None
    exp_19: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_99);  sub_99 = None
    sum_60: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_65: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_19, sum_60);  exp_19 = sum_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_213: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_65);  div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_118: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_213, [8, 16, 48, 48]);  clone_213 = None
    view_352: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_118, [128, 48, 48]);  expand_118 = None
    expand_119: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_175, [8, 16, 48, 784]);  getitem_175 = None
    clone_214: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
    view_353: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_214, [128, 48, 784]);  clone_214 = None
    bmm_39: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_352, view_353);  view_352 = view_353 = None
    view_354: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_39, [8, 16, 48, 784]);  bmm_39 = None
    permute_177: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_354, [0, 3, 1, 2]);  view_354 = None
    view_355: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_177, [8, 784, 768]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(arg496_1, [1, 0]);  arg496_1 = None
    clone_215: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_355, memory_format = torch.contiguous_format);  view_355 = None
    view_356: "f32[6272, 768]" = torch.ops.aten.view.default(clone_215, [6272, 768]);  clone_215 = None
    mm_19: "f32[6272, 768]" = torch.ops.aten.mm.default(view_356, permute_178);  view_356 = permute_178 = None
    view_357: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_19, [8, 784, 768]);  mm_19 = None
    add_282: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_357, arg497_1);  view_357 = arg497_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_216: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_282);  add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_385: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg76_1, clone_216);  arg76_1 = clone_216 = None
    add_283: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_279, mul_385);  add_279 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_217: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_283, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_217, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 784, 1]" = var_mean_58[0]
    getitem_177: "f32[8, 784, 1]" = var_mean_58[1];  var_mean_58 = None
    add_284: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_58: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
    sub_100: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_217, getitem_177);  clone_217 = getitem_177 = None
    mul_386: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_58);  sub_100 = rsqrt_58 = None
    mul_387: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_386, arg498_1);  mul_386 = arg498_1 = None
    add_285: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_387, arg499_1);  mul_387 = arg499_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_179: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_285, [0, 2, 1]);  add_285 = None
    view_358: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_179, [8, 768, 28, 28]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_42: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_358, arg500_1, arg501_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_358 = arg500_1 = arg501_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_388: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, 0.5)
    mul_389: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, 0.7071067811865476);  convolution_42 = None
    erf_40: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_389);  mul_389 = None
    add_286: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_390: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_388, add_286);  mul_388 = add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_50: "f32[768]" = torch.ops.prims.convert_element_type.default(arg694_1, torch.float32);  arg694_1 = None
    convert_element_type_51: "f32[768]" = torch.ops.prims.convert_element_type.default(arg695_1, torch.float32);  arg695_1 = None
    add_287: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_22: "f32[768]" = torch.ops.aten.sqrt.default(add_287);  add_287 = None
    reciprocal_22: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_391: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_183: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_184: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, -1);  unsqueeze_183 = None
    unsqueeze_185: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_391, -1);  mul_391 = None
    unsqueeze_186: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, -1);  unsqueeze_185 = None
    sub_101: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_390, unsqueeze_184);  mul_390 = unsqueeze_184 = None
    mul_392: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_186);  sub_101 = unsqueeze_186 = None
    unsqueeze_187: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
    unsqueeze_188: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, -1);  unsqueeze_187 = None
    mul_393: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_188);  mul_392 = unsqueeze_188 = None
    unsqueeze_189: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
    unsqueeze_190: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, -1);  unsqueeze_189 = None
    add_288: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_393, unsqueeze_190);  mul_393 = unsqueeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_43: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_288, arg504_1, arg505_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_288 = arg504_1 = arg505_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_359: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_43, [8, 768, 784]);  convolution_43 = None
    permute_180: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_394: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg78_1, permute_180);  arg78_1 = permute_180 = None
    add_289: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_283, mul_394);  add_283 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_218: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_289, memory_format = torch.contiguous_format)
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_218, [2], correction = 0, keepdim = True)
    getitem_178: "f32[8, 784, 1]" = var_mean_59[0]
    getitem_179: "f32[8, 784, 1]" = var_mean_59[1];  var_mean_59 = None
    add_290: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
    rsqrt_59: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
    sub_102: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_218, getitem_179);  clone_218 = getitem_179 = None
    mul_395: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_59);  sub_102 = rsqrt_59 = None
    mul_396: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_395, arg506_1);  mul_395 = arg506_1 = None
    add_291: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_396, arg507_1);  mul_396 = arg507_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_360: "f32[6272, 768]" = torch.ops.aten.view.default(add_291, [6272, 768]);  add_291 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(arg508_1, [1, 0]);  arg508_1 = None
    addmm_58: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg509_1, view_360, permute_181);  arg509_1 = view_360 = permute_181 = None
    view_361: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_58, [8, 784, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_397: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_361, 0.5)
    mul_398: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_361, 0.7071067811865476);  view_361 = None
    erf_41: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_398);  mul_398 = None
    add_292: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_399: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_397, add_292);  mul_397 = add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_219: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_399);  mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_362: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_219, [6272, 3072]);  clone_219 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
    addmm_59: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg511_1, view_362, permute_182);  arg511_1 = view_362 = permute_182 = None
    view_363: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_59, [8, 784, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_220: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_363);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_400: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg79_1, clone_220);  arg79_1 = clone_220 = None
    add_293: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_289, mul_400);  add_289 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_221: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_293, memory_format = torch.contiguous_format)
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_221, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 784, 1]" = var_mean_60[0]
    getitem_181: "f32[8, 784, 1]" = var_mean_60[1];  var_mean_60 = None
    add_294: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_60: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    sub_103: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_221, getitem_181);  clone_221 = getitem_181 = None
    mul_401: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_60);  sub_103 = rsqrt_60 = None
    mul_402: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_401, arg512_1);  mul_401 = arg512_1 = None
    add_295: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_402, arg513_1);  mul_402 = arg513_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_364: "f32[6272, 768]" = torch.ops.aten.view.default(add_295, [6272, 768]);  add_295 = None
    permute_183: "f32[768, 2304]" = torch.ops.aten.permute.default(arg514_1, [1, 0]);  arg514_1 = None
    addmm_60: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg515_1, view_364, permute_183);  arg515_1 = view_364 = permute_183 = None
    view_365: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_60, [8, 784, 2304]);  addmm_60 = None
    view_366: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_365, [8, 784, 3, 16, 48]);  view_365 = None
    permute_184: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_366, [2, 0, 3, 4, 1]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_184);  permute_184 = None
    getitem_182: "f32[8, 16, 48, 784]" = unbind_20[0]
    getitem_183: "f32[8, 16, 48, 784]" = unbind_20[1]
    getitem_184: "f32[8, 16, 48, 784]" = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_82: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_182, 2.0)
    sum_61: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_82, [-1], True);  pow_82 = None
    pow_83: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_61, 0.5);  sum_61 = None
    clamp_min_40: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_83, 1e-12);  pow_83 = None
    expand_120: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_40, [8, 16, 48, 784]);  clamp_min_40 = None
    div_66: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_182, expand_120);  getitem_182 = expand_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_84: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_183, 2.0)
    sum_62: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_84, [-1], True);  pow_84 = None
    pow_85: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_62, 0.5);  sum_62 = None
    clamp_min_41: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_85, 1e-12);  pow_85 = None
    expand_121: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_41, [8, 16, 48, 784]);  clamp_min_41 = None
    div_67: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_183, expand_121);  getitem_183 = expand_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_185: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_67, [0, 1, 3, 2]);  div_67 = None
    expand_122: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_66, [8, 16, 48, 784]);  div_66 = None
    clone_222: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
    view_367: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_222, [128, 48, 784]);  clone_222 = None
    expand_123: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_185, [8, 16, 784, 48]);  permute_185 = None
    clone_223: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
    view_368: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_223, [128, 784, 48]);  clone_223 = None
    bmm_40: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_367, view_368);  view_367 = view_368 = None
    view_369: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_40, [8, 16, 48, 48]);  bmm_40 = None
    mul_403: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_369, arg81_1);  view_369 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_403, [-1], True)
    sub_104: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_403, amax_20);  mul_403 = amax_20 = None
    exp_20: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_104);  sub_104 = None
    sum_63: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_68: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_20, sum_63);  exp_20 = sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_224: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_68);  div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_124: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_224, [8, 16, 48, 48]);  clone_224 = None
    view_370: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_124, [128, 48, 48]);  expand_124 = None
    expand_125: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_184, [8, 16, 48, 784]);  getitem_184 = None
    clone_225: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
    view_371: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_225, [128, 48, 784]);  clone_225 = None
    bmm_41: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_370, view_371);  view_370 = view_371 = None
    view_372: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_41, [8, 16, 48, 784]);  bmm_41 = None
    permute_186: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_372, [0, 3, 1, 2]);  view_372 = None
    view_373: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_186, [8, 784, 768]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(arg516_1, [1, 0]);  arg516_1 = None
    clone_226: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_373, memory_format = torch.contiguous_format);  view_373 = None
    view_374: "f32[6272, 768]" = torch.ops.aten.view.default(clone_226, [6272, 768]);  clone_226 = None
    mm_20: "f32[6272, 768]" = torch.ops.aten.mm.default(view_374, permute_187);  view_374 = permute_187 = None
    view_375: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_20, [8, 784, 768]);  mm_20 = None
    add_296: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_375, arg517_1);  view_375 = arg517_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_227: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_296);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_404: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg80_1, clone_227);  arg80_1 = clone_227 = None
    add_297: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_293, mul_404);  add_293 = mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_228: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_297, memory_format = torch.contiguous_format)
    var_mean_61 = torch.ops.aten.var_mean.correction(clone_228, [2], correction = 0, keepdim = True)
    getitem_185: "f32[8, 784, 1]" = var_mean_61[0]
    getitem_186: "f32[8, 784, 1]" = var_mean_61[1];  var_mean_61 = None
    add_298: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_185, 1e-06);  getitem_185 = None
    rsqrt_61: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
    sub_105: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_228, getitem_186);  clone_228 = getitem_186 = None
    mul_405: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_61);  sub_105 = rsqrt_61 = None
    mul_406: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_405, arg518_1);  mul_405 = arg518_1 = None
    add_299: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_406, arg519_1);  mul_406 = arg519_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_188: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_299, [0, 2, 1]);  add_299 = None
    view_376: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_188, [8, 768, 28, 28]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_44: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_376, arg520_1, arg521_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_376 = arg520_1 = arg521_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_407: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
    mul_408: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476);  convolution_44 = None
    erf_42: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
    add_300: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_409: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_407, add_300);  mul_407 = add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_52: "f32[768]" = torch.ops.prims.convert_element_type.default(arg697_1, torch.float32);  arg697_1 = None
    convert_element_type_53: "f32[768]" = torch.ops.prims.convert_element_type.default(arg698_1, torch.float32);  arg698_1 = None
    add_301: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_23: "f32[768]" = torch.ops.aten.sqrt.default(add_301);  add_301 = None
    reciprocal_23: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_410: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_191: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_192: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, -1);  unsqueeze_191 = None
    unsqueeze_193: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_410, -1);  mul_410 = None
    unsqueeze_194: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, -1);  unsqueeze_193 = None
    sub_106: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_409, unsqueeze_192);  mul_409 = unsqueeze_192 = None
    mul_411: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_194);  sub_106 = unsqueeze_194 = None
    unsqueeze_195: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
    unsqueeze_196: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, -1);  unsqueeze_195 = None
    mul_412: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_411, unsqueeze_196);  mul_411 = unsqueeze_196 = None
    unsqueeze_197: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg523_1, -1);  arg523_1 = None
    unsqueeze_198: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, -1);  unsqueeze_197 = None
    add_302: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_198);  mul_412 = unsqueeze_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_45: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_302, arg524_1, arg525_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_302 = arg524_1 = arg525_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_377: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_45, [8, 768, 784]);  convolution_45 = None
    permute_189: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_413: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg82_1, permute_189);  arg82_1 = permute_189 = None
    add_303: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_297, mul_413);  add_297 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_229: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_303, memory_format = torch.contiguous_format)
    var_mean_62 = torch.ops.aten.var_mean.correction(clone_229, [2], correction = 0, keepdim = True)
    getitem_187: "f32[8, 784, 1]" = var_mean_62[0]
    getitem_188: "f32[8, 784, 1]" = var_mean_62[1];  var_mean_62 = None
    add_304: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-06);  getitem_187 = None
    rsqrt_62: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_304);  add_304 = None
    sub_107: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_229, getitem_188);  clone_229 = getitem_188 = None
    mul_414: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_62);  sub_107 = rsqrt_62 = None
    mul_415: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_414, arg526_1);  mul_414 = arg526_1 = None
    add_305: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_415, arg527_1);  mul_415 = arg527_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_378: "f32[6272, 768]" = torch.ops.aten.view.default(add_305, [6272, 768]);  add_305 = None
    permute_190: "f32[768, 3072]" = torch.ops.aten.permute.default(arg528_1, [1, 0]);  arg528_1 = None
    addmm_61: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg529_1, view_378, permute_190);  arg529_1 = view_378 = permute_190 = None
    view_379: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_61, [8, 784, 3072]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_416: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_379, 0.5)
    mul_417: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_379, 0.7071067811865476);  view_379 = None
    erf_43: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_417);  mul_417 = None
    add_306: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_418: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_416, add_306);  mul_416 = add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_230: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_418);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_380: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_230, [6272, 3072]);  clone_230 = None
    permute_191: "f32[3072, 768]" = torch.ops.aten.permute.default(arg530_1, [1, 0]);  arg530_1 = None
    addmm_62: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg531_1, view_380, permute_191);  arg531_1 = view_380 = permute_191 = None
    view_381: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_62, [8, 784, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_231: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_381);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_419: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg83_1, clone_231);  arg83_1 = clone_231 = None
    add_307: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_303, mul_419);  add_303 = mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_232: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_307, memory_format = torch.contiguous_format)
    var_mean_63 = torch.ops.aten.var_mean.correction(clone_232, [2], correction = 0, keepdim = True)
    getitem_189: "f32[8, 784, 1]" = var_mean_63[0]
    getitem_190: "f32[8, 784, 1]" = var_mean_63[1];  var_mean_63 = None
    add_308: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_189, 1e-06);  getitem_189 = None
    rsqrt_63: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_308);  add_308 = None
    sub_108: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_232, getitem_190);  clone_232 = getitem_190 = None
    mul_420: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_63);  sub_108 = rsqrt_63 = None
    mul_421: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_420, arg532_1);  mul_420 = arg532_1 = None
    add_309: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_421, arg533_1);  mul_421 = arg533_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_382: "f32[6272, 768]" = torch.ops.aten.view.default(add_309, [6272, 768]);  add_309 = None
    permute_192: "f32[768, 2304]" = torch.ops.aten.permute.default(arg534_1, [1, 0]);  arg534_1 = None
    addmm_63: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg535_1, view_382, permute_192);  arg535_1 = view_382 = permute_192 = None
    view_383: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_63, [8, 784, 2304]);  addmm_63 = None
    view_384: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_383, [8, 784, 3, 16, 48]);  view_383 = None
    permute_193: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_384, [2, 0, 3, 4, 1]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_193);  permute_193 = None
    getitem_191: "f32[8, 16, 48, 784]" = unbind_21[0]
    getitem_192: "f32[8, 16, 48, 784]" = unbind_21[1]
    getitem_193: "f32[8, 16, 48, 784]" = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_86: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_191, 2.0)
    sum_64: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_86, [-1], True);  pow_86 = None
    pow_87: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_64, 0.5);  sum_64 = None
    clamp_min_42: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_87, 1e-12);  pow_87 = None
    expand_126: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_42, [8, 16, 48, 784]);  clamp_min_42 = None
    div_69: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_191, expand_126);  getitem_191 = expand_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_88: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_192, 2.0)
    sum_65: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_88, [-1], True);  pow_88 = None
    pow_89: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_65, 0.5);  sum_65 = None
    clamp_min_43: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_89, 1e-12);  pow_89 = None
    expand_127: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_43, [8, 16, 48, 784]);  clamp_min_43 = None
    div_70: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_192, expand_127);  getitem_192 = expand_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_194: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_70, [0, 1, 3, 2]);  div_70 = None
    expand_128: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_69, [8, 16, 48, 784]);  div_69 = None
    clone_233: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    view_385: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_233, [128, 48, 784]);  clone_233 = None
    expand_129: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_194, [8, 16, 784, 48]);  permute_194 = None
    clone_234: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    view_386: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_234, [128, 784, 48]);  clone_234 = None
    bmm_42: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_385, view_386);  view_385 = view_386 = None
    view_387: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_42, [8, 16, 48, 48]);  bmm_42 = None
    mul_422: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_387, arg85_1);  view_387 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_422, [-1], True)
    sub_109: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_422, amax_21);  mul_422 = amax_21 = None
    exp_21: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_109);  sub_109 = None
    sum_66: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_71: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_21, sum_66);  exp_21 = sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_235: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_71);  div_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_130: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_235, [8, 16, 48, 48]);  clone_235 = None
    view_388: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_130, [128, 48, 48]);  expand_130 = None
    expand_131: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_193, [8, 16, 48, 784]);  getitem_193 = None
    clone_236: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
    view_389: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_236, [128, 48, 784]);  clone_236 = None
    bmm_43: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_388, view_389);  view_388 = view_389 = None
    view_390: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_43, [8, 16, 48, 784]);  bmm_43 = None
    permute_195: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_390, [0, 3, 1, 2]);  view_390 = None
    view_391: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_195, [8, 784, 768]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(arg536_1, [1, 0]);  arg536_1 = None
    clone_237: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_391, memory_format = torch.contiguous_format);  view_391 = None
    view_392: "f32[6272, 768]" = torch.ops.aten.view.default(clone_237, [6272, 768]);  clone_237 = None
    mm_21: "f32[6272, 768]" = torch.ops.aten.mm.default(view_392, permute_196);  view_392 = permute_196 = None
    view_393: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_21, [8, 784, 768]);  mm_21 = None
    add_310: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_393, arg537_1);  view_393 = arg537_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_238: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_310);  add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_423: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg84_1, clone_238);  arg84_1 = clone_238 = None
    add_311: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_307, mul_423);  add_307 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_239: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_311, memory_format = torch.contiguous_format)
    var_mean_64 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
    getitem_194: "f32[8, 784, 1]" = var_mean_64[0]
    getitem_195: "f32[8, 784, 1]" = var_mean_64[1];  var_mean_64 = None
    add_312: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
    rsqrt_64: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_312);  add_312 = None
    sub_110: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_239, getitem_195);  clone_239 = getitem_195 = None
    mul_424: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_64);  sub_110 = rsqrt_64 = None
    mul_425: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_424, arg538_1);  mul_424 = arg538_1 = None
    add_313: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_425, arg539_1);  mul_425 = arg539_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_197: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_313, [0, 2, 1]);  add_313 = None
    view_394: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_197, [8, 768, 28, 28]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_46: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_394, arg540_1, arg541_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_394 = arg540_1 = arg541_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_426: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, 0.5)
    mul_427: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, 0.7071067811865476);  convolution_46 = None
    erf_44: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_427);  mul_427 = None
    add_314: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_428: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_426, add_314);  mul_426 = add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_54: "f32[768]" = torch.ops.prims.convert_element_type.default(arg700_1, torch.float32);  arg700_1 = None
    convert_element_type_55: "f32[768]" = torch.ops.prims.convert_element_type.default(arg701_1, torch.float32);  arg701_1 = None
    add_315: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_24: "f32[768]" = torch.ops.aten.sqrt.default(add_315);  add_315 = None
    reciprocal_24: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_429: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_199: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_200: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, -1);  unsqueeze_199 = None
    unsqueeze_201: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
    unsqueeze_202: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, -1);  unsqueeze_201 = None
    sub_111: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_428, unsqueeze_200);  mul_428 = unsqueeze_200 = None
    mul_430: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_202);  sub_111 = unsqueeze_202 = None
    unsqueeze_203: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg542_1, -1);  arg542_1 = None
    unsqueeze_204: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, -1);  unsqueeze_203 = None
    mul_431: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_204);  mul_430 = unsqueeze_204 = None
    unsqueeze_205: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg543_1, -1);  arg543_1 = None
    unsqueeze_206: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, -1);  unsqueeze_205 = None
    add_316: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_206);  mul_431 = unsqueeze_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_47: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_316, arg544_1, arg545_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_316 = arg544_1 = arg545_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_395: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_47, [8, 768, 784]);  convolution_47 = None
    permute_198: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_395, [0, 2, 1]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_432: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg86_1, permute_198);  arg86_1 = permute_198 = None
    add_317: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_311, mul_432);  add_311 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_240: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_317, memory_format = torch.contiguous_format)
    var_mean_65 = torch.ops.aten.var_mean.correction(clone_240, [2], correction = 0, keepdim = True)
    getitem_196: "f32[8, 784, 1]" = var_mean_65[0]
    getitem_197: "f32[8, 784, 1]" = var_mean_65[1];  var_mean_65 = None
    add_318: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-06);  getitem_196 = None
    rsqrt_65: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_318);  add_318 = None
    sub_112: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_240, getitem_197);  clone_240 = getitem_197 = None
    mul_433: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_65);  sub_112 = rsqrt_65 = None
    mul_434: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_433, arg546_1);  mul_433 = arg546_1 = None
    add_319: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_434, arg547_1);  mul_434 = arg547_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_396: "f32[6272, 768]" = torch.ops.aten.view.default(add_319, [6272, 768]);  add_319 = None
    permute_199: "f32[768, 3072]" = torch.ops.aten.permute.default(arg548_1, [1, 0]);  arg548_1 = None
    addmm_64: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg549_1, view_396, permute_199);  arg549_1 = view_396 = permute_199 = None
    view_397: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_64, [8, 784, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_435: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_397, 0.5)
    mul_436: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_397, 0.7071067811865476);  view_397 = None
    erf_45: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_436);  mul_436 = None
    add_320: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_437: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_435, add_320);  mul_435 = add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_241: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_437);  mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_398: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_241, [6272, 3072]);  clone_241 = None
    permute_200: "f32[3072, 768]" = torch.ops.aten.permute.default(arg550_1, [1, 0]);  arg550_1 = None
    addmm_65: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg551_1, view_398, permute_200);  arg551_1 = view_398 = permute_200 = None
    view_399: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_65, [8, 784, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_242: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_399);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_438: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg87_1, clone_242);  arg87_1 = clone_242 = None
    add_321: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_317, mul_438);  add_317 = mul_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_243: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_321, memory_format = torch.contiguous_format)
    var_mean_66 = torch.ops.aten.var_mean.correction(clone_243, [2], correction = 0, keepdim = True)
    getitem_198: "f32[8, 784, 1]" = var_mean_66[0]
    getitem_199: "f32[8, 784, 1]" = var_mean_66[1];  var_mean_66 = None
    add_322: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
    rsqrt_66: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_322);  add_322 = None
    sub_113: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_243, getitem_199);  clone_243 = getitem_199 = None
    mul_439: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_66);  sub_113 = rsqrt_66 = None
    mul_440: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_439, arg552_1);  mul_439 = arg552_1 = None
    add_323: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_440, arg553_1);  mul_440 = arg553_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_400: "f32[6272, 768]" = torch.ops.aten.view.default(add_323, [6272, 768]);  add_323 = None
    permute_201: "f32[768, 2304]" = torch.ops.aten.permute.default(arg554_1, [1, 0]);  arg554_1 = None
    addmm_66: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg555_1, view_400, permute_201);  arg555_1 = view_400 = permute_201 = None
    view_401: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_66, [8, 784, 2304]);  addmm_66 = None
    view_402: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_401, [8, 784, 3, 16, 48]);  view_401 = None
    permute_202: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_402, [2, 0, 3, 4, 1]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_202);  permute_202 = None
    getitem_200: "f32[8, 16, 48, 784]" = unbind_22[0]
    getitem_201: "f32[8, 16, 48, 784]" = unbind_22[1]
    getitem_202: "f32[8, 16, 48, 784]" = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_90: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_200, 2.0)
    sum_67: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_90, [-1], True);  pow_90 = None
    pow_91: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_67, 0.5);  sum_67 = None
    clamp_min_44: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_91, 1e-12);  pow_91 = None
    expand_132: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_44, [8, 16, 48, 784]);  clamp_min_44 = None
    div_72: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_200, expand_132);  getitem_200 = expand_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_92: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_201, 2.0)
    sum_68: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_92, [-1], True);  pow_92 = None
    pow_93: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_68, 0.5);  sum_68 = None
    clamp_min_45: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_93, 1e-12);  pow_93 = None
    expand_133: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_45, [8, 16, 48, 784]);  clamp_min_45 = None
    div_73: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_201, expand_133);  getitem_201 = expand_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_203: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_73, [0, 1, 3, 2]);  div_73 = None
    expand_134: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_72, [8, 16, 48, 784]);  div_72 = None
    clone_244: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
    view_403: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_244, [128, 48, 784]);  clone_244 = None
    expand_135: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_203, [8, 16, 784, 48]);  permute_203 = None
    clone_245: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
    view_404: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_245, [128, 784, 48]);  clone_245 = None
    bmm_44: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_403, view_404);  view_403 = view_404 = None
    view_405: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_44, [8, 16, 48, 48]);  bmm_44 = None
    mul_441: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_405, arg89_1);  view_405 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_22: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_441, [-1], True)
    sub_114: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_441, amax_22);  mul_441 = amax_22 = None
    exp_22: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
    sum_69: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_74: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_22, sum_69);  exp_22 = sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_246: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_74);  div_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_136: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_246, [8, 16, 48, 48]);  clone_246 = None
    view_406: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_136, [128, 48, 48]);  expand_136 = None
    expand_137: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_202, [8, 16, 48, 784]);  getitem_202 = None
    clone_247: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    view_407: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_247, [128, 48, 784]);  clone_247 = None
    bmm_45: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_406, view_407);  view_406 = view_407 = None
    view_408: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_45, [8, 16, 48, 784]);  bmm_45 = None
    permute_204: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_408, [0, 3, 1, 2]);  view_408 = None
    view_409: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_204, [8, 784, 768]);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(arg556_1, [1, 0]);  arg556_1 = None
    clone_248: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_409, memory_format = torch.contiguous_format);  view_409 = None
    view_410: "f32[6272, 768]" = torch.ops.aten.view.default(clone_248, [6272, 768]);  clone_248 = None
    mm_22: "f32[6272, 768]" = torch.ops.aten.mm.default(view_410, permute_205);  view_410 = permute_205 = None
    view_411: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_22, [8, 784, 768]);  mm_22 = None
    add_324: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_411, arg557_1);  view_411 = arg557_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_249: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_324);  add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_442: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg88_1, clone_249);  arg88_1 = clone_249 = None
    add_325: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_321, mul_442);  add_321 = mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_250: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_325, memory_format = torch.contiguous_format)
    var_mean_67 = torch.ops.aten.var_mean.correction(clone_250, [2], correction = 0, keepdim = True)
    getitem_203: "f32[8, 784, 1]" = var_mean_67[0]
    getitem_204: "f32[8, 784, 1]" = var_mean_67[1];  var_mean_67 = None
    add_326: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_203, 1e-06);  getitem_203 = None
    rsqrt_67: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
    sub_115: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_250, getitem_204);  clone_250 = getitem_204 = None
    mul_443: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_67);  sub_115 = rsqrt_67 = None
    mul_444: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_443, arg558_1);  mul_443 = arg558_1 = None
    add_327: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_444, arg559_1);  mul_444 = arg559_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_206: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_327, [0, 2, 1]);  add_327 = None
    view_412: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_206, [8, 768, 28, 28]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_48: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_412, arg560_1, arg561_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_412 = arg560_1 = arg561_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_445: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, 0.5)
    mul_446: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, 0.7071067811865476);  convolution_48 = None
    erf_46: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_446);  mul_446 = None
    add_328: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_447: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_445, add_328);  mul_445 = add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_56: "f32[768]" = torch.ops.prims.convert_element_type.default(arg703_1, torch.float32);  arg703_1 = None
    convert_element_type_57: "f32[768]" = torch.ops.prims.convert_element_type.default(arg704_1, torch.float32);  arg704_1 = None
    add_329: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_25: "f32[768]" = torch.ops.aten.sqrt.default(add_329);  add_329 = None
    reciprocal_25: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_448: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_207: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_208: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, -1);  unsqueeze_207 = None
    unsqueeze_209: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_448, -1);  mul_448 = None
    unsqueeze_210: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, -1);  unsqueeze_209 = None
    sub_116: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_447, unsqueeze_208);  mul_447 = unsqueeze_208 = None
    mul_449: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_210);  sub_116 = unsqueeze_210 = None
    unsqueeze_211: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg562_1, -1);  arg562_1 = None
    unsqueeze_212: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, -1);  unsqueeze_211 = None
    mul_450: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_449, unsqueeze_212);  mul_449 = unsqueeze_212 = None
    unsqueeze_213: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg563_1, -1);  arg563_1 = None
    unsqueeze_214: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, -1);  unsqueeze_213 = None
    add_330: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_450, unsqueeze_214);  mul_450 = unsqueeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_49: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_330, arg564_1, arg565_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_330 = arg564_1 = arg565_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_413: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_49, [8, 768, 784]);  convolution_49 = None
    permute_207: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_451: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg90_1, permute_207);  arg90_1 = permute_207 = None
    add_331: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_325, mul_451);  add_325 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_251: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_331, memory_format = torch.contiguous_format)
    var_mean_68 = torch.ops.aten.var_mean.correction(clone_251, [2], correction = 0, keepdim = True)
    getitem_205: "f32[8, 784, 1]" = var_mean_68[0]
    getitem_206: "f32[8, 784, 1]" = var_mean_68[1];  var_mean_68 = None
    add_332: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_205, 1e-06);  getitem_205 = None
    rsqrt_68: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_332);  add_332 = None
    sub_117: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_251, getitem_206);  clone_251 = getitem_206 = None
    mul_452: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_68);  sub_117 = rsqrt_68 = None
    mul_453: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_452, arg566_1);  mul_452 = arg566_1 = None
    add_333: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_453, arg567_1);  mul_453 = arg567_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_414: "f32[6272, 768]" = torch.ops.aten.view.default(add_333, [6272, 768]);  add_333 = None
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(arg568_1, [1, 0]);  arg568_1 = None
    addmm_67: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg569_1, view_414, permute_208);  arg569_1 = view_414 = permute_208 = None
    view_415: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_67, [8, 784, 3072]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_454: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
    mul_455: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
    erf_47: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_455);  mul_455 = None
    add_334: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_456: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_454, add_334);  mul_454 = add_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_252: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_456);  mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_416: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_252, [6272, 3072]);  clone_252 = None
    permute_209: "f32[3072, 768]" = torch.ops.aten.permute.default(arg570_1, [1, 0]);  arg570_1 = None
    addmm_68: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg571_1, view_416, permute_209);  arg571_1 = view_416 = permute_209 = None
    view_417: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_68, [8, 784, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_253: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_417);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_457: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg91_1, clone_253);  arg91_1 = clone_253 = None
    add_335: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_331, mul_457);  add_331 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_254: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_335, memory_format = torch.contiguous_format)
    var_mean_69 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
    getitem_207: "f32[8, 784, 1]" = var_mean_69[0]
    getitem_208: "f32[8, 784, 1]" = var_mean_69[1];  var_mean_69 = None
    add_336: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_207, 1e-06);  getitem_207 = None
    rsqrt_69: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
    sub_118: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_254, getitem_208);  clone_254 = getitem_208 = None
    mul_458: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_69);  sub_118 = rsqrt_69 = None
    mul_459: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_458, arg572_1);  mul_458 = arg572_1 = None
    add_337: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_459, arg573_1);  mul_459 = arg573_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_418: "f32[6272, 768]" = torch.ops.aten.view.default(add_337, [6272, 768]);  add_337 = None
    permute_210: "f32[768, 2304]" = torch.ops.aten.permute.default(arg574_1, [1, 0]);  arg574_1 = None
    addmm_69: "f32[6272, 2304]" = torch.ops.aten.addmm.default(arg575_1, view_418, permute_210);  arg575_1 = view_418 = permute_210 = None
    view_419: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_69, [8, 784, 2304]);  addmm_69 = None
    view_420: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_419, [8, 784, 3, 16, 48]);  view_419 = None
    permute_211: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_420, [2, 0, 3, 4, 1]);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_211);  permute_211 = None
    getitem_209: "f32[8, 16, 48, 784]" = unbind_23[0]
    getitem_210: "f32[8, 16, 48, 784]" = unbind_23[1]
    getitem_211: "f32[8, 16, 48, 784]" = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_94: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_209, 2.0)
    sum_70: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_94, [-1], True);  pow_94 = None
    pow_95: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_70, 0.5);  sum_70 = None
    clamp_min_46: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_95, 1e-12);  pow_95 = None
    expand_138: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_46, [8, 16, 48, 784]);  clamp_min_46 = None
    div_75: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_209, expand_138);  getitem_209 = expand_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_96: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_210, 2.0)
    sum_71: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_96, [-1], True);  pow_96 = None
    pow_97: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_71, 0.5);  sum_71 = None
    clamp_min_47: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_97, 1e-12);  pow_97 = None
    expand_139: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_47, [8, 16, 48, 784]);  clamp_min_47 = None
    div_76: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_210, expand_139);  getitem_210 = expand_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_212: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_76, [0, 1, 3, 2]);  div_76 = None
    expand_140: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_75, [8, 16, 48, 784]);  div_75 = None
    clone_255: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
    view_421: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_255, [128, 48, 784]);  clone_255 = None
    expand_141: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_212, [8, 16, 784, 48]);  permute_212 = None
    clone_256: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
    view_422: "f32[128, 784, 48]" = torch.ops.aten.view.default(clone_256, [128, 784, 48]);  clone_256 = None
    bmm_46: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_421, view_422);  view_421 = view_422 = None
    view_423: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_46, [8, 16, 48, 48]);  bmm_46 = None
    mul_460: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_423, arg93_1);  view_423 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_23: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_460, [-1], True)
    sub_119: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_460, amax_23);  mul_460 = amax_23 = None
    exp_23: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_119);  sub_119 = None
    sum_72: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_77: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_23, sum_72);  exp_23 = sum_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_257: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_77);  div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_142: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_257, [8, 16, 48, 48]);  clone_257 = None
    view_424: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_142, [128, 48, 48]);  expand_142 = None
    expand_143: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_211, [8, 16, 48, 784]);  getitem_211 = None
    clone_258: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
    view_425: "f32[128, 48, 784]" = torch.ops.aten.view.default(clone_258, [128, 48, 784]);  clone_258 = None
    bmm_47: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_424, view_425);  view_424 = view_425 = None
    view_426: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_47, [8, 16, 48, 784]);  bmm_47 = None
    permute_213: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_426, [0, 3, 1, 2]);  view_426 = None
    view_427: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_213, [8, 784, 768]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(arg576_1, [1, 0]);  arg576_1 = None
    clone_259: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_427, memory_format = torch.contiguous_format);  view_427 = None
    view_428: "f32[6272, 768]" = torch.ops.aten.view.default(clone_259, [6272, 768]);  clone_259 = None
    mm_23: "f32[6272, 768]" = torch.ops.aten.mm.default(view_428, permute_214);  view_428 = permute_214 = None
    view_429: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_23, [8, 784, 768]);  mm_23 = None
    add_338: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_429, arg577_1);  view_429 = arg577_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_260: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_338);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_461: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg92_1, clone_260);  arg92_1 = clone_260 = None
    add_339: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_335, mul_461);  add_335 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_261: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_339, memory_format = torch.contiguous_format)
    var_mean_70 = torch.ops.aten.var_mean.correction(clone_261, [2], correction = 0, keepdim = True)
    getitem_212: "f32[8, 784, 1]" = var_mean_70[0]
    getitem_213: "f32[8, 784, 1]" = var_mean_70[1];  var_mean_70 = None
    add_340: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-06);  getitem_212 = None
    rsqrt_70: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_340);  add_340 = None
    sub_120: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_261, getitem_213);  clone_261 = getitem_213 = None
    mul_462: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_70);  sub_120 = rsqrt_70 = None
    mul_463: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_462, arg578_1);  mul_462 = arg578_1 = None
    add_341: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_463, arg579_1);  mul_463 = arg579_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_215: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_341, [0, 2, 1]);  add_341 = None
    view_430: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_215, [8, 768, 28, 28]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_50: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_430, arg580_1, arg581_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_430 = arg580_1 = arg581_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_464: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, 0.5)
    mul_465: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, 0.7071067811865476);  convolution_50 = None
    erf_48: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_465);  mul_465 = None
    add_342: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_466: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_464, add_342);  mul_464 = add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    convert_element_type_58: "f32[768]" = torch.ops.prims.convert_element_type.default(arg706_1, torch.float32);  arg706_1 = None
    convert_element_type_59: "f32[768]" = torch.ops.prims.convert_element_type.default(arg707_1, torch.float32);  arg707_1 = None
    add_343: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_26: "f32[768]" = torch.ops.aten.sqrt.default(add_343);  add_343 = None
    reciprocal_26: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_467: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_215: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_216: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, -1);  unsqueeze_215 = None
    unsqueeze_217: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_467, -1);  mul_467 = None
    unsqueeze_218: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, -1);  unsqueeze_217 = None
    sub_121: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_466, unsqueeze_216);  mul_466 = unsqueeze_216 = None
    mul_468: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_218);  sub_121 = unsqueeze_218 = None
    unsqueeze_219: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg582_1, -1);  arg582_1 = None
    unsqueeze_220: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, -1);  unsqueeze_219 = None
    mul_469: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_468, unsqueeze_220);  mul_468 = unsqueeze_220 = None
    unsqueeze_221: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg583_1, -1);  arg583_1 = None
    unsqueeze_222: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, -1);  unsqueeze_221 = None
    add_344: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_469, unsqueeze_222);  mul_469 = unsqueeze_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_51: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_344, arg584_1, arg585_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_344 = arg584_1 = arg585_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_431: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_51, [8, 768, 784]);  convolution_51 = None
    permute_216: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_470: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg94_1, permute_216);  arg94_1 = permute_216 = None
    add_345: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_339, mul_470);  add_339 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_262: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_345, memory_format = torch.contiguous_format)
    var_mean_71 = torch.ops.aten.var_mean.correction(clone_262, [2], correction = 0, keepdim = True)
    getitem_214: "f32[8, 784, 1]" = var_mean_71[0]
    getitem_215: "f32[8, 784, 1]" = var_mean_71[1];  var_mean_71 = None
    add_346: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
    rsqrt_71: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
    sub_122: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_262, getitem_215);  clone_262 = getitem_215 = None
    mul_471: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_71);  sub_122 = rsqrt_71 = None
    mul_472: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_471, arg586_1);  mul_471 = arg586_1 = None
    add_347: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_472, arg587_1);  mul_472 = arg587_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_432: "f32[6272, 768]" = torch.ops.aten.view.default(add_347, [6272, 768]);  add_347 = None
    permute_217: "f32[768, 3072]" = torch.ops.aten.permute.default(arg588_1, [1, 0]);  arg588_1 = None
    addmm_70: "f32[6272, 3072]" = torch.ops.aten.addmm.default(arg589_1, view_432, permute_217);  arg589_1 = view_432 = permute_217 = None
    view_433: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_70, [8, 784, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_473: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_433, 0.5)
    mul_474: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476);  view_433 = None
    erf_49: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_474);  mul_474 = None
    add_348: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_475: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_473, add_348);  mul_473 = add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_263: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(mul_475);  mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_434: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_263, [6272, 3072]);  clone_263 = None
    permute_218: "f32[3072, 768]" = torch.ops.aten.permute.default(arg590_1, [1, 0]);  arg590_1 = None
    addmm_71: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg591_1, view_434, permute_218);  arg591_1 = view_434 = permute_218 = None
    view_435: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_71, [8, 784, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_264: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_435);  view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_476: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(arg95_1, clone_264);  arg95_1 = clone_264 = None
    add_349: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_345, mul_476);  add_345 = mul_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:447, code: x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
    expand_144: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg96_1, [8, -1, -1]);  arg96_1 = None
    cat_3: "f32[8, 785, 768]" = torch.ops.aten.cat.default([expand_144, add_349], 1);  expand_144 = add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    var_mean_72 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_216: "f32[8, 785, 1]" = var_mean_72[0]
    getitem_217: "f32[8, 785, 1]" = var_mean_72[1];  var_mean_72 = None
    add_350: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-06);  getitem_216 = None
    rsqrt_72: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_350);  add_350 = None
    sub_123: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(cat_3, getitem_217);  getitem_217 = None
    mul_477: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_72);  sub_123 = rsqrt_72 = None
    mul_478: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_477, arg592_1);  mul_477 = arg592_1 = None
    add_351: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_478, arg593_1);  mul_478 = arg593_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_29: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_351, 0, 0, 9223372036854775807)
    select: "f32[8, 768]" = torch.ops.aten.select.int(slice_29, 1, 0);  slice_29 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(arg594_1, [1, 0]);  arg594_1 = None
    addmm_72: "f32[8, 768]" = torch.ops.aten.addmm.default(arg595_1, select, permute_219);  arg595_1 = select = permute_219 = None
    unsqueeze_223: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_72, 1);  addmm_72 = None
    view_436: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_223, [8, 1, 16, 48]);  unsqueeze_223 = None
    permute_220: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_437: "f32[6280, 768]" = torch.ops.aten.view.default(add_351, [6280, 768])
    permute_221: "f32[768, 768]" = torch.ops.aten.permute.default(arg596_1, [1, 0]);  arg596_1 = None
    addmm_73: "f32[6280, 768]" = torch.ops.aten.addmm.default(arg597_1, view_437, permute_221);  arg597_1 = view_437 = permute_221 = None
    view_438: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_73, [8, 785, 768]);  addmm_73 = None
    view_439: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_438, [8, 785, 16, 48]);  view_438 = None
    permute_222: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_439, [0, 2, 1, 3]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_440: "f32[6280, 768]" = torch.ops.aten.view.default(add_351, [6280, 768])
    permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(arg598_1, [1, 0]);  arg598_1 = None
    addmm_74: "f32[6280, 768]" = torch.ops.aten.addmm.default(arg599_1, view_440, permute_223);  arg599_1 = view_440 = permute_223 = None
    view_441: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_74, [8, 785, 768]);  addmm_74 = None
    view_442: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_441, [8, 785, 16, 48]);  view_441 = None
    permute_224: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_220, permute_222, permute_224);  permute_220 = permute_222 = permute_224 = None
    getitem_218: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_225: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
    view_443: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_225, [8, 1, 768]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_444: "f32[8, 768]" = torch.ops.aten.view.default(view_443, [8, 768]);  view_443 = None
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(arg600_1, [1, 0]);  arg600_1 = None
    addmm_75: "f32[8, 768]" = torch.ops.aten.addmm.default(arg601_1, view_444, permute_226);  arg601_1 = view_444 = permute_226 = None
    view_445: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_75, [8, 1, 768]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_265: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_445);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_30: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_351, 0, 0, 9223372036854775807);  add_351 = None
    slice_31: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_30, 1, 1, 9223372036854775807);  slice_30 = None
    cat_4: "f32[8, 785, 768]" = torch.ops.aten.cat.default([clone_265, slice_31], 1);  clone_265 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_479: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(arg97_1, cat_4);  arg97_1 = cat_4 = None
    add_352: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(cat_3, mul_479);  cat_3 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    var_mean_73 = torch.ops.aten.var_mean.correction(add_352, [2], correction = 0, keepdim = True)
    getitem_227: "f32[8, 785, 1]" = var_mean_73[0]
    getitem_228: "f32[8, 785, 1]" = var_mean_73[1];  var_mean_73 = None
    add_353: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_227, 1e-06);  getitem_227 = None
    rsqrt_73: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_353);  add_353 = None
    sub_124: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_352, getitem_228);  add_352 = getitem_228 = None
    mul_480: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_73);  sub_124 = rsqrt_73 = None
    mul_481: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_480, arg602_1);  mul_480 = arg602_1 = None
    add_354: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_481, arg603_1);  mul_481 = arg603_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_32: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_354, 0, 0, 9223372036854775807)
    slice_33: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 1);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_227: "f32[768, 3072]" = torch.ops.aten.permute.default(arg604_1, [1, 0]);  arg604_1 = None
    view_446: "f32[8, 768]" = torch.ops.aten.view.default(slice_33, [8, 768]);  slice_33 = None
    mm_24: "f32[8, 3072]" = torch.ops.aten.mm.default(view_446, permute_227);  view_446 = permute_227 = None
    view_447: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_24, [8, 1, 3072]);  mm_24 = None
    add_355: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_447, arg605_1);  view_447 = arg605_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_482: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_355, 0.5)
    mul_483: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_355, 0.7071067811865476);  add_355 = None
    erf_50: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_483);  mul_483 = None
    add_356: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    mul_484: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_482, add_356);  mul_482 = add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_266: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(mul_484);  mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_448: "f32[8, 3072]" = torch.ops.aten.view.default(clone_266, [8, 3072]);  clone_266 = None
    permute_228: "f32[3072, 768]" = torch.ops.aten.permute.default(arg606_1, [1, 0]);  arg606_1 = None
    addmm_76: "f32[8, 768]" = torch.ops.aten.addmm.default(arg607_1, view_448, permute_228);  arg607_1 = view_448 = permute_228 = None
    view_449: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_76, [8, 1, 768]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_267: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_449);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_485: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg98_1, clone_267);  arg98_1 = clone_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_34: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_354, 0, 0, 9223372036854775807)
    slice_35: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_34, 1, 1, 9223372036854775807);  slice_34 = None
    cat_5: "f32[8, 785, 768]" = torch.ops.aten.cat.default([mul_485, slice_35], 1);  mul_485 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    add_357: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_354, cat_5);  add_354 = cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    var_mean_74 = torch.ops.aten.var_mean.correction(add_357, [2], correction = 0, keepdim = True)
    getitem_229: "f32[8, 785, 1]" = var_mean_74[0]
    getitem_230: "f32[8, 785, 1]" = var_mean_74[1];  var_mean_74 = None
    add_358: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_229, 1e-06);  getitem_229 = None
    rsqrt_74: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_358);  add_358 = None
    sub_125: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_357, getitem_230);  getitem_230 = None
    mul_486: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_74);  sub_125 = rsqrt_74 = None
    mul_487: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_486, arg608_1);  mul_486 = arg608_1 = None
    add_359: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_487, arg609_1);  mul_487 = arg609_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_36: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_359, 0, 0, 9223372036854775807)
    select_1: "f32[8, 768]" = torch.ops.aten.select.int(slice_36, 1, 0);  slice_36 = None
    permute_229: "f32[768, 768]" = torch.ops.aten.permute.default(arg610_1, [1, 0]);  arg610_1 = None
    addmm_77: "f32[8, 768]" = torch.ops.aten.addmm.default(arg611_1, select_1, permute_229);  arg611_1 = select_1 = permute_229 = None
    unsqueeze_224: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_77, 1);  addmm_77 = None
    view_450: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_224, [8, 1, 16, 48]);  unsqueeze_224 = None
    permute_230: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_451: "f32[6280, 768]" = torch.ops.aten.view.default(add_359, [6280, 768])
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(arg612_1, [1, 0]);  arg612_1 = None
    addmm_78: "f32[6280, 768]" = torch.ops.aten.addmm.default(arg613_1, view_451, permute_231);  arg613_1 = view_451 = permute_231 = None
    view_452: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_78, [8, 785, 768]);  addmm_78 = None
    view_453: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_452, [8, 785, 16, 48]);  view_452 = None
    permute_232: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_454: "f32[6280, 768]" = torch.ops.aten.view.default(add_359, [6280, 768])
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(arg614_1, [1, 0]);  arg614_1 = None
    addmm_79: "f32[6280, 768]" = torch.ops.aten.addmm.default(arg615_1, view_454, permute_233);  arg615_1 = view_454 = permute_233 = None
    view_455: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_79, [8, 785, 768]);  addmm_79 = None
    view_456: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_455, [8, 785, 16, 48]);  view_455 = None
    permute_234: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_230, permute_232, permute_234);  permute_230 = permute_232 = permute_234 = None
    getitem_231: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_235: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_231, [0, 2, 1, 3]);  getitem_231 = None
    view_457: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_235, [8, 1, 768]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_458: "f32[8, 768]" = torch.ops.aten.view.default(view_457, [8, 768]);  view_457 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(arg616_1, [1, 0]);  arg616_1 = None
    addmm_80: "f32[8, 768]" = torch.ops.aten.addmm.default(arg617_1, view_458, permute_236);  arg617_1 = view_458 = permute_236 = None
    view_459: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_80, [8, 1, 768]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_268: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_459);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_37: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_359, 0, 0, 9223372036854775807);  add_359 = None
    slice_38: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_37, 1, 1, 9223372036854775807);  slice_37 = None
    cat_6: "f32[8, 785, 768]" = torch.ops.aten.cat.default([clone_268, slice_38], 1);  clone_268 = slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_488: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(arg99_1, cat_6);  arg99_1 = cat_6 = None
    add_360: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_357, mul_488);  add_357 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    var_mean_75 = torch.ops.aten.var_mean.correction(add_360, [2], correction = 0, keepdim = True)
    getitem_240: "f32[8, 785, 1]" = var_mean_75[0]
    getitem_241: "f32[8, 785, 1]" = var_mean_75[1];  var_mean_75 = None
    add_361: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-06);  getitem_240 = None
    rsqrt_75: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
    sub_126: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_360, getitem_241);  add_360 = getitem_241 = None
    mul_489: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_75);  sub_126 = rsqrt_75 = None
    mul_490: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_489, arg618_1);  mul_489 = arg618_1 = None
    add_362: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_490, arg619_1);  mul_490 = arg619_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_39: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_362, 0, 0, 9223372036854775807)
    slice_40: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_39, 1, 0, 1);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_237: "f32[768, 3072]" = torch.ops.aten.permute.default(arg620_1, [1, 0]);  arg620_1 = None
    view_460: "f32[8, 768]" = torch.ops.aten.view.default(slice_40, [8, 768]);  slice_40 = None
    mm_25: "f32[8, 3072]" = torch.ops.aten.mm.default(view_460, permute_237);  view_460 = permute_237 = None
    view_461: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_25, [8, 1, 3072]);  mm_25 = None
    add_363: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_461, arg621_1);  view_461 = arg621_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_491: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_363, 0.5)
    mul_492: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_363, 0.7071067811865476);  add_363 = None
    erf_51: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_492);  mul_492 = None
    add_364: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    mul_493: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_491, add_364);  mul_491 = add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_269: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(mul_493);  mul_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_462: "f32[8, 3072]" = torch.ops.aten.view.default(clone_269, [8, 3072]);  clone_269 = None
    permute_238: "f32[3072, 768]" = torch.ops.aten.permute.default(arg622_1, [1, 0]);  arg622_1 = None
    addmm_81: "f32[8, 768]" = torch.ops.aten.addmm.default(arg623_1, view_462, permute_238);  arg623_1 = view_462 = permute_238 = None
    view_463: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_81, [8, 1, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_270: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_463);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_494: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg100_1, clone_270);  arg100_1 = clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_41: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_362, 0, 0, 9223372036854775807)
    slice_42: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_41, 1, 1, 9223372036854775807);  slice_41 = None
    cat_7: "f32[8, 785, 768]" = torch.ops.aten.cat.default([mul_494, slice_42], 1);  mul_494 = slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    add_365: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_362, cat_7);  add_362 = cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:455, code: x = self.norm(x)
    var_mean_76 = torch.ops.aten.var_mean.correction(add_365, [2], correction = 0, keepdim = True)
    getitem_242: "f32[8, 785, 1]" = var_mean_76[0]
    getitem_243: "f32[8, 785, 1]" = var_mean_76[1];  var_mean_76 = None
    add_366: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
    rsqrt_76: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
    sub_127: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_365, getitem_243);  add_365 = getitem_243 = None
    mul_495: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_76);  sub_127 = rsqrt_76 = None
    mul_496: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_495, arg624_1);  mul_495 = arg624_1 = None
    add_367: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_496, arg625_1);  mul_496 = arg625_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:460, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_43: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_367, 0, 0, 9223372036854775807);  add_367 = None
    select_2: "f32[8, 768]" = torch.ops.aten.select.int(slice_43, 1, 0);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:461, code: x = self.head_drop(x)
    clone_271: "f32[8, 768]" = torch.ops.aten.clone.default(select_2);  select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:462, code: return x if pre_logits else self.head(x)
    permute_239: "f32[768, 1000]" = torch.ops.aten.permute.default(arg626_1, [1, 0]);  arg626_1 = None
    addmm_82: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg627_1, clone_271, permute_239);  arg627_1 = clone_271 = permute_239 = None
    return (addmm_82,)
    