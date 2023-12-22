from __future__ import annotations



def forward(self, arg0_1: "f32[128, 2560]", arg1_1: "f32[128, 2560]", arg2_1: "f32[8008, 2560]", arg3_1: "f32[2560]", arg4_1: "f32[2560]", arg5_1: "f32[2560, 2560]", arg6_1: "f32[2560]", arg7_1: "f32[2560, 2560]", arg8_1: "f32[2560]", arg9_1: "f32[2560, 2560]", arg10_1: "f32[2560]", arg11_1: "f32[2560, 2560]", arg12_1: "f32[2560]", arg13_1: "f32[2560]", arg14_1: "f32[2560]", arg15_1: "f32[10240, 2560]", arg16_1: "f32[10240]", arg17_1: "f32[2560, 10240]", arg18_1: "f32[2560]", arg19_1: "f32[2560]", arg20_1: "f32[2560]", arg21_1: "f32[2560, 2560]", arg22_1: "f32[2560]", arg23_1: "f32[2560, 2560]", arg24_1: "f32[2560]", arg25_1: "f32[2560, 2560]", arg26_1: "f32[2560]", arg27_1: "f32[2560, 2560]", arg28_1: "f32[2560]", arg29_1: "f32[2560]", arg30_1: "f32[2560]", arg31_1: "f32[10240, 2560]", arg32_1: "f32[10240]", arg33_1: "f32[2560, 10240]", arg34_1: "f32[2560]", arg35_1: "f32[2560]", arg36_1: "f32[2560]", arg37_1: "f32[2560]", arg38_1: "f32[2560]", arg39_1: "f32[2560, 2560]", arg40_1: "f32[2560]", arg41_1: "f32[2560, 2560]", arg42_1: "f32[2560]", arg43_1: "f32[2560, 2560]", arg44_1: "f32[2560]", arg45_1: "f32[2560, 2560]", arg46_1: "f32[2560]", arg47_1: "f32[2560]", arg48_1: "f32[2560]", arg49_1: "f32[2560, 2560]", arg50_1: "f32[2560]", arg51_1: "f32[2560, 2560]", arg52_1: "f32[2560]", arg53_1: "f32[2560, 2560]", arg54_1: "f32[2560]", arg55_1: "f32[2560, 2560]", arg56_1: "f32[2560]", arg57_1: "f32[2560]", arg58_1: "f32[2560]", arg59_1: "f32[10240, 2560]", arg60_1: "f32[10240]", arg61_1: "f32[2560, 10240]", arg62_1: "f32[2560]", arg63_1: "f32[2560]", arg64_1: "f32[2560]", arg65_1: "f32[2560, 2560]", arg66_1: "f32[2560]", arg67_1: "f32[2560, 2560]", arg68_1: "f32[2560]", arg69_1: "f32[2560, 2560]", arg70_1: "f32[2560]", arg71_1: "f32[2560, 2560]", arg72_1: "f32[2560]", arg73_1: "f32[2560]", arg74_1: "f32[2560]", arg75_1: "f32[2560, 2560]", arg76_1: "f32[2560]", arg77_1: "f32[2560, 2560]", arg78_1: "f32[2560]", arg79_1: "f32[2560, 2560]", arg80_1: "f32[2560]", arg81_1: "f32[2560, 2560]", arg82_1: "f32[2560]", arg83_1: "f32[2560]", arg84_1: "f32[2560]", arg85_1: "f32[10240, 2560]", arg86_1: "f32[10240]", arg87_1: "f32[2560, 10240]", arg88_1: "f32[2560]", arg89_1: "f32[2560]", arg90_1: "f32[2560]", arg91_1: "f32[2560, 2560]", arg92_1: "f32[2560]", arg93_1: "f32[2560, 2560]", arg94_1: "f32[2560]", arg95_1: "f32[2560, 2560]", arg96_1: "f32[2560]", arg97_1: "f32[2560, 2560]", arg98_1: "f32[2560]", arg99_1: "f32[2560]", arg100_1: "f32[2560]", arg101_1: "f32[2560, 2560]", arg102_1: "f32[2560]", arg103_1: "f32[2560, 2560]", arg104_1: "f32[2560]", arg105_1: "f32[2560, 2560]", arg106_1: "f32[2560]", arg107_1: "f32[2560, 2560]", arg108_1: "f32[2560]", arg109_1: "f32[2560]", arg110_1: "f32[2560]", arg111_1: "f32[10240, 2560]", arg112_1: "f32[10240]", arg113_1: "f32[2560, 10240]", arg114_1: "f32[2560]", arg115_1: "f32[2560]", arg116_1: "f32[2560]", arg117_1: "f32[2560, 2560]", arg118_1: "f32[2560]", arg119_1: "f32[2560, 2560]", arg120_1: "f32[2560]", arg121_1: "f32[2560, 2560]", arg122_1: "f32[2560]", arg123_1: "f32[2560, 2560]", arg124_1: "f32[2560]", arg125_1: "f32[2560]", arg126_1: "f32[2560]", arg127_1: "f32[2560, 2560]", arg128_1: "f32[2560]", arg129_1: "f32[2560, 2560]", arg130_1: "f32[2560]", arg131_1: "f32[2560, 2560]", arg132_1: "f32[2560]", arg133_1: "f32[2560, 2560]", arg134_1: "f32[2560]", arg135_1: "f32[2560]", arg136_1: "f32[2560]", arg137_1: "f32[10240, 2560]", arg138_1: "f32[10240]", arg139_1: "f32[2560, 10240]", arg140_1: "f32[2560]", arg141_1: "f32[2560]", arg142_1: "f32[2560]", arg143_1: "f32[2560, 2560]", arg144_1: "f32[2560]", arg145_1: "f32[2560, 2560]", arg146_1: "f32[2560]", arg147_1: "f32[2560, 2560]", arg148_1: "f32[2560]", arg149_1: "f32[2560, 2560]", arg150_1: "f32[2560]", arg151_1: "f32[2560]", arg152_1: "f32[2560]", arg153_1: "f32[2560, 2560]", arg154_1: "f32[2560]", arg155_1: "f32[2560, 2560]", arg156_1: "f32[2560]", arg157_1: "f32[2560, 2560]", arg158_1: "f32[2560]", arg159_1: "f32[2560, 2560]", arg160_1: "f32[2560]", arg161_1: "f32[2560]", arg162_1: "f32[2560]", arg163_1: "f32[10240, 2560]", arg164_1: "f32[10240]", arg165_1: "f32[2560, 10240]", arg166_1: "f32[2560]", arg167_1: "f32[2560]", arg168_1: "f32[2560]", arg169_1: "f32[2560, 2560]", arg170_1: "f32[2560]", arg171_1: "f32[2560, 2560]", arg172_1: "f32[2560]", arg173_1: "f32[2560, 2560]", arg174_1: "f32[2560]", arg175_1: "f32[2560, 2560]", arg176_1: "f32[2560]", arg177_1: "f32[2560]", arg178_1: "f32[2560]", arg179_1: "f32[2560, 2560]", arg180_1: "f32[2560]", arg181_1: "f32[2560, 2560]", arg182_1: "f32[2560]", arg183_1: "f32[2560, 2560]", arg184_1: "f32[2560]", arg185_1: "f32[2560, 2560]", arg186_1: "f32[2560]", arg187_1: "f32[2560]", arg188_1: "f32[2560]", arg189_1: "f32[10240, 2560]", arg190_1: "f32[10240]", arg191_1: "f32[2560, 10240]", arg192_1: "f32[2560]", arg193_1: "f32[2560]", arg194_1: "f32[2560]", arg195_1: "f32[2560, 2560]", arg196_1: "f32[2560]", arg197_1: "f32[2560, 2560]", arg198_1: "f32[2560]", arg199_1: "f32[2560, 2560]", arg200_1: "f32[2560]", arg201_1: "f32[2560, 2560]", arg202_1: "f32[2560]", arg203_1: "f32[2560]", arg204_1: "f32[2560]", arg205_1: "f32[2560, 2560]", arg206_1: "f32[2560]", arg207_1: "f32[2560, 2560]", arg208_1: "f32[2560]", arg209_1: "f32[2560, 2560]", arg210_1: "f32[2560]", arg211_1: "f32[2560, 2560]", arg212_1: "f32[2560]", arg213_1: "f32[2560]", arg214_1: "f32[2560]", arg215_1: "f32[10240, 2560]", arg216_1: "f32[10240]", arg217_1: "f32[2560, 10240]", arg218_1: "f32[2560]", arg219_1: "f32[2560]", arg220_1: "f32[2560]", arg221_1: "f32[2560, 2560]", arg222_1: "f32[2560]", arg223_1: "f32[2560, 2560]", arg224_1: "f32[2560]", arg225_1: "f32[2560, 2560]", arg226_1: "f32[2560]", arg227_1: "f32[2560, 2560]", arg228_1: "f32[2560]", arg229_1: "f32[2560]", arg230_1: "f32[2560]", arg231_1: "f32[2560, 2560]", arg232_1: "f32[2560]", arg233_1: "f32[2560, 2560]", arg234_1: "f32[2560]", arg235_1: "f32[2560, 2560]", arg236_1: "f32[2560]", arg237_1: "f32[2560, 2560]", arg238_1: "f32[2560]", arg239_1: "f32[2560]", arg240_1: "f32[2560]", arg241_1: "f32[10240, 2560]", arg242_1: "f32[10240]", arg243_1: "f32[2560, 10240]", arg244_1: "f32[2560]", arg245_1: "f32[2560]", arg246_1: "f32[2560]", arg247_1: "f32[2560, 2560]", arg248_1: "f32[2560]", arg249_1: "f32[2560, 2560]", arg250_1: "f32[2560]", arg251_1: "f32[2560, 2560]", arg252_1: "f32[2560]", arg253_1: "f32[2560, 2560]", arg254_1: "f32[2560]", arg255_1: "f32[2560]", arg256_1: "f32[2560]", arg257_1: "f32[2560, 2560]", arg258_1: "f32[2560]", arg259_1: "f32[2560, 2560]", arg260_1: "f32[2560]", arg261_1: "f32[2560, 2560]", arg262_1: "f32[2560]", arg263_1: "f32[2560, 2560]", arg264_1: "f32[2560]", arg265_1: "f32[2560]", arg266_1: "f32[2560]", arg267_1: "f32[10240, 2560]", arg268_1: "f32[10240]", arg269_1: "f32[2560, 10240]", arg270_1: "f32[2560]", arg271_1: "f32[2560]", arg272_1: "f32[2560]", arg273_1: "f32[2560, 2560]", arg274_1: "f32[2560]", arg275_1: "f32[2560, 2560]", arg276_1: "f32[2560]", arg277_1: "f32[2560, 2560]", arg278_1: "f32[2560]", arg279_1: "f32[2560, 2560]", arg280_1: "f32[2560]", arg281_1: "f32[2560]", arg282_1: "f32[2560]", arg283_1: "f32[2560, 2560]", arg284_1: "f32[2560]", arg285_1: "f32[2560, 2560]", arg286_1: "f32[2560]", arg287_1: "f32[2560, 2560]", arg288_1: "f32[2560]", arg289_1: "f32[2560, 2560]", arg290_1: "f32[2560]", arg291_1: "f32[2560]", arg292_1: "f32[2560]", arg293_1: "f32[10240, 2560]", arg294_1: "f32[10240]", arg295_1: "f32[2560, 10240]", arg296_1: "f32[2560]", arg297_1: "f32[2560]", arg298_1: "f32[2560]", arg299_1: "f32[2560, 2560]", arg300_1: "f32[2560]", arg301_1: "f32[2560, 2560]", arg302_1: "f32[2560]", arg303_1: "f32[2560, 2560]", arg304_1: "f32[2560]", arg305_1: "f32[2560, 2560]", arg306_1: "f32[2560]", arg307_1: "f32[2560]", arg308_1: "f32[2560]", arg309_1: "f32[2560, 2560]", arg310_1: "f32[2560]", arg311_1: "f32[2560, 2560]", arg312_1: "f32[2560]", arg313_1: "f32[2560, 2560]", arg314_1: "f32[2560]", arg315_1: "f32[2560, 2560]", arg316_1: "f32[2560]", arg317_1: "f32[2560]", arg318_1: "f32[2560]", arg319_1: "f32[10240, 2560]", arg320_1: "f32[10240]", arg321_1: "f32[2560, 10240]", arg322_1: "f32[2560]", arg323_1: "f32[2560]", arg324_1: "f32[2560]", arg325_1: "f32[2560, 2560]", arg326_1: "f32[2560]", arg327_1: "f32[2560, 2560]", arg328_1: "f32[2560]", arg329_1: "f32[2560, 2560]", arg330_1: "f32[2560]", arg331_1: "f32[2560, 2560]", arg332_1: "f32[2560]", arg333_1: "f32[2560]", arg334_1: "f32[2560]", arg335_1: "f32[2560, 2560]", arg336_1: "f32[2560]", arg337_1: "f32[2560, 2560]", arg338_1: "f32[2560]", arg339_1: "f32[2560, 2560]", arg340_1: "f32[2560]", arg341_1: "f32[2560, 2560]", arg342_1: "f32[2560]", arg343_1: "f32[2560]", arg344_1: "f32[2560]", arg345_1: "f32[10240, 2560]", arg346_1: "f32[10240]", arg347_1: "f32[2560, 10240]", arg348_1: "f32[2560]", arg349_1: "f32[2560]", arg350_1: "f32[2560]", arg351_1: "f32[2560, 2560]", arg352_1: "f32[2560]", arg353_1: "f32[2560, 2560]", arg354_1: "f32[2560]", arg355_1: "f32[2560, 2560]", arg356_1: "f32[2560]", arg357_1: "f32[2560, 2560]", arg358_1: "f32[2560]", arg359_1: "f32[2560]", arg360_1: "f32[2560]", arg361_1: "f32[2560, 2560]", arg362_1: "f32[2560]", arg363_1: "f32[2560, 2560]", arg364_1: "f32[2560]", arg365_1: "f32[2560, 2560]", arg366_1: "f32[2560]", arg367_1: "f32[2560, 2560]", arg368_1: "f32[2560]", arg369_1: "f32[2560]", arg370_1: "f32[2560]", arg371_1: "f32[10240, 2560]", arg372_1: "f32[10240]", arg373_1: "f32[2560, 10240]", arg374_1: "f32[2560]", arg375_1: "f32[2560]", arg376_1: "f32[2560]", arg377_1: "f32[2560, 2560]", arg378_1: "f32[2560]", arg379_1: "f32[2560, 2560]", arg380_1: "f32[2560]", arg381_1: "f32[2560, 2560]", arg382_1: "f32[2560]", arg383_1: "f32[2560, 2560]", arg384_1: "f32[2560]", arg385_1: "f32[2560]", arg386_1: "f32[2560]", arg387_1: "f32[2560, 2560]", arg388_1: "f32[2560]", arg389_1: "f32[2560, 2560]", arg390_1: "f32[2560]", arg391_1: "f32[2560, 2560]", arg392_1: "f32[2560]", arg393_1: "f32[2560, 2560]", arg394_1: "f32[2560]", arg395_1: "f32[2560]", arg396_1: "f32[2560]", arg397_1: "f32[10240, 2560]", arg398_1: "f32[10240]", arg399_1: "f32[2560, 10240]", arg400_1: "f32[2560]", arg401_1: "f32[2560]", arg402_1: "f32[2560]", arg403_1: "f32[2560, 2560]", arg404_1: "f32[2560]", arg405_1: "f32[2560, 2560]", arg406_1: "f32[2560]", arg407_1: "f32[2560, 2560]", arg408_1: "f32[2560]", arg409_1: "f32[2560, 2560]", arg410_1: "f32[2560]", arg411_1: "f32[2560]", arg412_1: "f32[2560]", arg413_1: "f32[2560, 2560]", arg414_1: "f32[2560]", arg415_1: "f32[2560, 2560]", arg416_1: "f32[2560]", arg417_1: "f32[2560, 2560]", arg418_1: "f32[2560]", arg419_1: "f32[2560, 2560]", arg420_1: "f32[2560]", arg421_1: "f32[2560]", arg422_1: "f32[2560]", arg423_1: "f32[10240, 2560]", arg424_1: "f32[10240]", arg425_1: "f32[2560, 10240]", arg426_1: "f32[2560]", arg427_1: "f32[2560]", arg428_1: "f32[2560]", arg429_1: "f32[2560, 2560]", arg430_1: "f32[2560]", arg431_1: "f32[2560, 2560]", arg432_1: "f32[2560]", arg433_1: "f32[2560, 2560]", arg434_1: "f32[2560]", arg435_1: "f32[2560, 2560]", arg436_1: "f32[2560]", arg437_1: "f32[2560]", arg438_1: "f32[2560]", arg439_1: "f32[2560, 2560]", arg440_1: "f32[2560]", arg441_1: "f32[2560, 2560]", arg442_1: "f32[2560]", arg443_1: "f32[2560, 2560]", arg444_1: "f32[2560]", arg445_1: "f32[2560, 2560]", arg446_1: "f32[2560]", arg447_1: "f32[2560]", arg448_1: "f32[2560]", arg449_1: "f32[10240, 2560]", arg450_1: "f32[10240]", arg451_1: "f32[2560, 10240]", arg452_1: "f32[2560]", arg453_1: "f32[2560]", arg454_1: "f32[2560]", arg455_1: "f32[2560, 2560]", arg456_1: "f32[2560]", arg457_1: "f32[2560, 2560]", arg458_1: "f32[2560]", arg459_1: "f32[2560, 2560]", arg460_1: "f32[2560]", arg461_1: "f32[2560, 2560]", arg462_1: "f32[2560]", arg463_1: "f32[2560]", arg464_1: "f32[2560]", arg465_1: "f32[2560, 2560]", arg466_1: "f32[2560]", arg467_1: "f32[2560, 2560]", arg468_1: "f32[2560]", arg469_1: "f32[2560, 2560]", arg470_1: "f32[2560]", arg471_1: "f32[2560, 2560]", arg472_1: "f32[2560]", arg473_1: "f32[2560]", arg474_1: "f32[2560]", arg475_1: "f32[10240, 2560]", arg476_1: "f32[10240]", arg477_1: "f32[2560, 10240]", arg478_1: "f32[2560]", arg479_1: "f32[2560]", arg480_1: "f32[2560]", arg481_1: "f32[2560, 2560]", arg482_1: "f32[2560]", arg483_1: "f32[2560, 2560]", arg484_1: "f32[2560]", arg485_1: "f32[2560, 2560]", arg486_1: "f32[2560]", arg487_1: "f32[2560, 2560]", arg488_1: "f32[2560]", arg489_1: "f32[2560]", arg490_1: "f32[2560]", arg491_1: "f32[2560, 2560]", arg492_1: "f32[2560]", arg493_1: "f32[2560, 2560]", arg494_1: "f32[2560]", arg495_1: "f32[2560, 2560]", arg496_1: "f32[2560]", arg497_1: "f32[2560, 2560]", arg498_1: "f32[2560]", arg499_1: "f32[2560]", arg500_1: "f32[2560]", arg501_1: "f32[10240, 2560]", arg502_1: "f32[10240]", arg503_1: "f32[2560, 10240]", arg504_1: "f32[2560]", arg505_1: "f32[2560]", arg506_1: "f32[2560]", arg507_1: "f32[2560, 2560]", arg508_1: "f32[2560]", arg509_1: "f32[2560, 2560]", arg510_1: "f32[2560]", arg511_1: "f32[2560, 2560]", arg512_1: "f32[2560]", arg513_1: "f32[2560, 2560]", arg514_1: "f32[2560]", arg515_1: "f32[2560]", arg516_1: "f32[2560]", arg517_1: "f32[2560, 2560]", arg518_1: "f32[2560]", arg519_1: "f32[2560, 2560]", arg520_1: "f32[2560]", arg521_1: "f32[2560, 2560]", arg522_1: "f32[2560]", arg523_1: "f32[2560, 2560]", arg524_1: "f32[2560]", arg525_1: "f32[2560]", arg526_1: "f32[2560]", arg527_1: "f32[10240, 2560]", arg528_1: "f32[10240]", arg529_1: "f32[2560, 10240]", arg530_1: "f32[2560]", arg531_1: "f32[2560]", arg532_1: "f32[2560]", arg533_1: "f32[2560, 2560]", arg534_1: "f32[2560]", arg535_1: "f32[2560, 2560]", arg536_1: "f32[2560]", arg537_1: "f32[2560, 2560]", arg538_1: "f32[2560]", arg539_1: "f32[2560, 2560]", arg540_1: "f32[2560]", arg541_1: "f32[2560]", arg542_1: "f32[2560]", arg543_1: "f32[2560, 2560]", arg544_1: "f32[2560]", arg545_1: "f32[2560, 2560]", arg546_1: "f32[2560]", arg547_1: "f32[2560, 2560]", arg548_1: "f32[2560]", arg549_1: "f32[2560, 2560]", arg550_1: "f32[2560]", arg551_1: "f32[2560]", arg552_1: "f32[2560]", arg553_1: "f32[10240, 2560]", arg554_1: "f32[10240]", arg555_1: "f32[2560, 10240]", arg556_1: "f32[2560]", arg557_1: "f32[2560]", arg558_1: "f32[2560]", arg559_1: "f32[2560, 2560]", arg560_1: "f32[2560]", arg561_1: "f32[2560, 2560]", arg562_1: "f32[2560]", arg563_1: "f32[2560, 2560]", arg564_1: "f32[2560]", arg565_1: "f32[2560, 2560]", arg566_1: "f32[2560]", arg567_1: "f32[2560]", arg568_1: "f32[2560]", arg569_1: "f32[2560, 2560]", arg570_1: "f32[2560]", arg571_1: "f32[2560, 2560]", arg572_1: "f32[2560]", arg573_1: "f32[2560, 2560]", arg574_1: "f32[2560]", arg575_1: "f32[2560, 2560]", arg576_1: "f32[2560]", arg577_1: "f32[2560]", arg578_1: "f32[2560]", arg579_1: "f32[10240, 2560]", arg580_1: "f32[10240]", arg581_1: "f32[2560, 10240]", arg582_1: "f32[2560]", arg583_1: "f32[2560]", arg584_1: "f32[2560]", arg585_1: "f32[2560, 2560]", arg586_1: "f32[2560]", arg587_1: "f32[2560, 2560]", arg588_1: "f32[2560]", arg589_1: "f32[2560, 2560]", arg590_1: "f32[2560]", arg591_1: "f32[2560, 2560]", arg592_1: "f32[2560]", arg593_1: "f32[2560]", arg594_1: "f32[2560]", arg595_1: "f32[2560, 2560]", arg596_1: "f32[2560]", arg597_1: "f32[2560, 2560]", arg598_1: "f32[2560]", arg599_1: "f32[2560, 2560]", arg600_1: "f32[2560]", arg601_1: "f32[2560, 2560]", arg602_1: "f32[2560]", arg603_1: "f32[2560]", arg604_1: "f32[2560]", arg605_1: "f32[10240, 2560]", arg606_1: "f32[10240]", arg607_1: "f32[2560, 10240]", arg608_1: "f32[2560]", arg609_1: "f32[2560]", arg610_1: "f32[2560]", arg611_1: "f32[2560, 2560]", arg612_1: "f32[2560]", arg613_1: "f32[2560, 2560]", arg614_1: "f32[2560]", arg615_1: "f32[2560, 2560]", arg616_1: "f32[2560]", arg617_1: "f32[2560, 2560]", arg618_1: "f32[2560]", arg619_1: "f32[2560]", arg620_1: "f32[2560]", arg621_1: "f32[2560, 2560]", arg622_1: "f32[2560]", arg623_1: "f32[2560, 2560]", arg624_1: "f32[2560]", arg625_1: "f32[2560, 2560]", arg626_1: "f32[2560]", arg627_1: "f32[2560, 2560]", arg628_1: "f32[2560]", arg629_1: "f32[2560]", arg630_1: "f32[2560]", arg631_1: "f32[10240, 2560]", arg632_1: "f32[10240]", arg633_1: "f32[2560, 10240]", arg634_1: "f32[2560]", arg635_1: "f32[2560]", arg636_1: "f32[2560]", arg637_1: "f32[2560, 2560]", arg638_1: "f32[2560]", arg639_1: "f32[2560, 2560]", arg640_1: "f32[2560]", arg641_1: "f32[2560, 2560]", arg642_1: "f32[2560]", arg643_1: "f32[2560, 2560]", arg644_1: "f32[2560]", arg645_1: "f32[2560]", arg646_1: "f32[2560]", arg647_1: "f32[2560, 2560]", arg648_1: "f32[2560]", arg649_1: "f32[2560, 2560]", arg650_1: "f32[2560]", arg651_1: "f32[2560, 2560]", arg652_1: "f32[2560]", arg653_1: "f32[2560, 2560]", arg654_1: "f32[2560]", arg655_1: "f32[2560]", arg656_1: "f32[2560]", arg657_1: "f32[10240, 2560]", arg658_1: "f32[10240]", arg659_1: "f32[2560, 10240]", arg660_1: "f32[2560]", arg661_1: "f32[2560]", arg662_1: "f32[2560]", arg663_1: "f32[8008, 2560]", arg664_1: "f32[1, 8008]", arg665_1: "i64[1, 128]", arg666_1: "i64[1, 128]", arg667_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:737, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(arg667_1, [-1, 128]);  arg667_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:744, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 128, 2560]" = torch.ops.aten.embedding.default(arg2_1, view, 0);  view = None
    mul: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:122, code: positions = torch.arange(
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg0_1, iota);  arg0_1 = iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:748, code: hidden_states = inputs_embeds + embed_pos
    add: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    sub: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul_1: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_2: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_2, [128, 2560])
    permute: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg6_1, view_1, permute);  arg6_1 = view_1 = permute = None
    view_2: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm, [1, 128, 2560]);  addmm = None
    mul_3: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_2, 0.11180339887498948);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_3, [1, 128, 32, 80]);  mul_3 = None
    permute_5: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_3: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_10: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_3, [32, -1, 80]);  clone_3 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_75: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_2, [128, 2560])
    permute_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm_1: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg8_1, view_3, permute_1);  arg8_1 = view_3 = permute_1 = None
    view_4: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 2560]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_4, [1, -1, 32, 80]);  view_4 = None
    permute_2: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    clone_1: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_11: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_1, [32, -1, 80]);  clone_1 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_76: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_2, [128, 2560]);  add_2 = None
    permute_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_2: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg10_1, view_6, permute_3);  arg10_1 = view_6 = permute_3 = None
    view_7: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 2560]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_7, [1, -1, 32, 80]);  view_7 = None
    permute_4: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_12: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_2, [32, -1, 80]);  clone_2 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_77: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
    _scaled_dot_product_flash_attention_default_25 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_75, unsqueeze_default_76, unsqueeze_default_77, scale = 1.0);  unsqueeze_default_75 = unsqueeze_default_76 = unsqueeze_default_77 = None
    getitem_181: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_25[0];  _scaled_dot_product_flash_attention_default_25 = None
    squeeze_dim_25: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_181, 0);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_13: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_25, [1, 32, 128, 80]);  squeeze_dim_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_14: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_5, [1, 128, 2560]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_15: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_14, [128, 2560]);  view_14 = None
    permute_8: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    addmm_3: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg12_1, view_15, permute_8);  arg12_1 = view_15 = permute_8 = None
    view_16: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 2560]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:328, code: hidden_states = residual + hidden_states
    add_3: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add, view_16);  add = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  getitem_3 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_4: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
    add_5: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:332, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_5, [128, 2560]);  add_5 = None
    permute_9: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    addmm_4: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg16_1, view_17, permute_9);  arg16_1 = view_17 = permute_9 = None
    view_18: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 10240]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_6: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:334, code: hidden_states = self.fc2(hidden_states)
    view_19: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_8, [128, 10240]);  mul_8 = None
    permute_10: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    addmm_5: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg18_1, view_19, permute_10);  arg18_1 = view_19 = permute_10 = None
    view_20: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 2560]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:336, code: hidden_states = residual + hidden_states
    add_7: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_3, view_20);  add_3 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  getitem_5 = None
    add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    mul_9: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
    add_9: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_21: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_9, [128, 2560])
    permute_11: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    addmm_6: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg22_1, view_21, permute_11);  arg22_1 = view_21 = permute_11 = None
    view_22: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 2560]);  addmm_6 = None
    mul_11: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_22, 0.11180339887498948);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_11, [1, 128, 32, 80]);  mul_11 = None
    permute_16: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_11: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_30: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_11, [32, -1, 80]);  clone_11 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_72: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_23: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_9, [128, 2560])
    permute_12: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_7: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg24_1, view_23, permute_12);  arg24_1 = view_23 = permute_12 = None
    view_24: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 2560]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_24, [1, -1, 32, 80]);  view_24 = None
    permute_13: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_9: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_31: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_9, [32, -1, 80]);  clone_9 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_73: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_26: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_9, [128, 2560]);  add_9 = None
    permute_14: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_8: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg26_1, view_26, permute_14);  arg26_1 = view_26 = permute_14 = None
    view_27: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 2560]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_27, [1, -1, 32, 80]);  view_27 = None
    permute_15: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_10: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_32: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_10, [32, -1, 80]);  clone_10 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_74: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
    _scaled_dot_product_flash_attention_default_24 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_72, unsqueeze_default_73, unsqueeze_default_74, scale = 1.0);  unsqueeze_default_72 = unsqueeze_default_73 = unsqueeze_default_74 = None
    getitem_180: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_24[0];  _scaled_dot_product_flash_attention_default_24 = None
    squeeze_dim_24: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_180, 0);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_33: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_24, [1, 32, 128, 80]);  squeeze_dim_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_34: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_13, [1, 128, 2560]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_35: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_34, [128, 2560]);  view_34 = None
    permute_19: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_9: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg28_1, view_35, permute_19);  arg28_1 = view_35 = permute_19 = None
    view_36: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 2560]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:328, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_7, view_36);  add_7 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  getitem_7 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_12: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
    add_12: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:332, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_37: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_12, [128, 2560]);  add_12 = None
    permute_20: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    addmm_10: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg32_1, view_37, permute_20);  arg32_1 = view_37 = permute_20 = None
    view_38: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 10240]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_15: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_13: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_14, add_13);  mul_14 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:334, code: hidden_states = self.fc2(hidden_states)
    view_39: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_16, [128, 10240]);  mul_16 = None
    permute_21: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    addmm_11: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg34_1, view_39, permute_21);  arg34_1 = view_39 = permute_21 = None
    view_40: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 2560]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:336, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_10, view_40);  add_10 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:807, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:975, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_41: "i64[1, 128]" = torch.ops.aten.reshape.default(arg666_1, [-1, 128]);  arg666_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:985, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding_2: "f32[1, 128, 2560]" = torch.ops.aten.embedding.default(arg2_1, view_41, 0);  arg2_1 = view_41 = None
    mul_19: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:122, code: positions = torch.arange(
    iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg1_1, iota_2);  arg1_1 = iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:999, code: hidden_states = inputs_embeds + positions
    add_18: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_19, embedding_3);  mul_19 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_7: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_18, getitem_11);  getitem_11 = None
    add_19: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    mul_20: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
    mul_21: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_20, arg37_1);  mul_20 = arg37_1 = None
    add_20: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_21, arg38_1);  mul_21 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_43: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_20, [128, 2560])
    permute_22: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_12: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg40_1, view_43, permute_22);  arg40_1 = view_43 = permute_22 = None
    view_44: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 2560]);  addmm_12 = None
    mul_22: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_44, 0.11180339887498948);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_51: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_22, [1, 128, 32, 80]);  mul_22 = None
    permute_27: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    clone_20: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_52: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_20, [32, -1, 80]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_45: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_20, [128, 2560])
    permute_23: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_13: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg42_1, view_45, permute_23);  arg42_1 = view_45 = permute_23 = None
    view_46: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 2560]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_47: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_46, [1, -1, 32, 80]);  view_46 = None
    permute_24: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    clone_18: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_53: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_18, [32, -1, 80]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_4: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_52, permute_28);  view_52 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_55: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 32, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:87, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:88, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_17: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_42: "i64[128, 1]" = torch.ops.aten.reshape.default(add_17, [128, 1]);  add_17 = None
    lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota_1, view_42);  iota_1 = view_42 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:86, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:88, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where: "f32[128, 128]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_1: "f32[1, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 1, 128, 128]);  unsqueeze_3 = None
    add_21: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_55, expand_1);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_56: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_21, [32, 128, 128]);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_56, [-1], True)
    sub_8: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_56, amax_2);  view_56 = amax_2 = None
    exp_2: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_48: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_20, [128, 2560]);  add_20 = None
    permute_25: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    addmm_14: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg44_1, view_48, permute_25);  arg44_1 = view_48 = permute_25 = None
    view_49: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 2560]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_49, [1, -1, 32, 80]);  view_49 = None
    permute_26: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_19: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_54: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_19, [32, -1, 80]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_2, view_54);  div_2 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_57: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_5, [1, 32, 128, 80]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_22: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_58: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_22, [1, 128, 2560]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_59: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_58, [128, 2560]);  view_58 = None
    permute_30: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    addmm_15: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg46_1, view_59, permute_30);  arg46_1 = view_59 = permute_30 = None
    view_60: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 2560]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_22: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_18, view_60);  add_18 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_22, getitem_13);  getitem_13 = None
    add_23: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    mul_23: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_24: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_23, arg47_1);  mul_23 = arg47_1 = None
    add_24: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_24, arg48_1);  mul_24 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_61: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_24, [128, 2560]);  add_24 = None
    permute_31: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    addmm_16: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg50_1, view_61, permute_31);  arg50_1 = view_61 = permute_31 = None
    view_62: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 2560]);  addmm_16 = None
    mul_25: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_62, 0.11180339887498948);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_25, [1, 128, 32, 80]);  mul_25 = None
    permute_36: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_26: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_70: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_26, [32, -1, 80]);  clone_26 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_69: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_70, 0);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:807, code: hidden_states = self.layer_norm(hidden_states)
    sub_6: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    mul_17: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
    add_16: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_63: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_32: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    addmm_17: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg52_1, view_63, permute_32);  arg52_1 = view_63 = permute_32 = None
    view_64: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 2560]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_64, [1, -1, 32, 80]);  view_64 = None
    permute_33: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_24: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_71: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_24, [32, -1, 80]);  clone_24 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_70: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_66: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_34: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    addmm_18: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg54_1, view_66, permute_34);  arg54_1 = view_66 = permute_34 = None
    view_67: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 2560]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_67, [1, -1, 32, 80]);  view_67 = None
    permute_35: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_25: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_72: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_25, [32, -1, 80]);  clone_25 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_71: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
    _scaled_dot_product_flash_attention_default_23 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_69, unsqueeze_default_70, unsqueeze_default_71, scale = 1.0);  unsqueeze_default_69 = unsqueeze_default_70 = unsqueeze_default_71 = None
    getitem_179: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_23[0];  _scaled_dot_product_flash_attention_default_23 = None
    squeeze_dim_23: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_179, 0);  getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_73: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_23, [1, 32, 128, 80]);  squeeze_dim_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_38: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_28: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_74: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_28, [1, 128, 2560]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_75: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_74, [128, 2560]);  view_74 = None
    permute_39: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_19: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg56_1, view_75, permute_39);  arg56_1 = view_75 = permute_39 = None
    view_76: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 2560]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_25: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_22, view_76);  add_22 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_25, getitem_15);  getitem_15 = None
    add_26: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    mul_26: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_27: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_26, arg57_1);  mul_26 = arg57_1 = None
    add_27: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_27, arg58_1);  mul_27 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_77: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_27, [128, 2560]);  add_27 = None
    permute_40: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    addmm_20: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg60_1, view_77, permute_40);  arg60_1 = view_77 = permute_40 = None
    view_78: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 10240]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_28: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_29: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_2: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
    add_28: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_30: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_28, add_28);  mul_28 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_79: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_30, [128, 10240]);  mul_30 = None
    permute_41: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    addmm_21: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg62_1, view_79, permute_41);  arg62_1 = view_79 = permute_41 = None
    view_80: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 2560]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_29: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_25, view_80);  add_25 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_29, getitem_17);  getitem_17 = None
    add_30: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    mul_31: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_32: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_31, arg63_1);  mul_31 = arg63_1 = None
    add_31: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_32, arg64_1);  mul_32 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_81: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_31, [128, 2560])
    permute_42: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    addmm_22: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg66_1, view_81, permute_42);  arg66_1 = view_81 = permute_42 = None
    view_82: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 2560]);  addmm_22 = None
    mul_33: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_82, 0.11180339887498948);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_33, [1, 128, 32, 80]);  mul_33 = None
    permute_47: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_34: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_90: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_34, [32, -1, 80]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_83: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_31, [128, 2560])
    permute_43: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    addmm_23: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg68_1, view_83, permute_43);  arg68_1 = view_83 = permute_43 = None
    view_84: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 2560]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_84, [1, -1, 32, 80]);  view_84 = None
    permute_44: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    clone_32: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_91: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_32, [32, -1, 80]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_48: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    bmm_8: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_90, permute_48);  view_90 = permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_93: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 32, 128, 128]);  bmm_8 = None
    add_32: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_93, expand_1);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_94: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_32, [32, 128, 128]);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_94, [-1], True)
    sub_13: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_94, amax_4);  view_94 = amax_4 = None
    exp_4: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_86: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_31, [128, 2560]);  add_31 = None
    permute_45: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    addmm_24: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg70_1, view_86, permute_45);  arg70_1 = view_86 = permute_45 = None
    view_87: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 2560]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_87, [1, -1, 32, 80]);  view_87 = None
    permute_46: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
    clone_33: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_92: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_33, [32, -1, 80]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_4, view_92);  div_4 = view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_95: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_9, [1, 32, 128, 80]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_49: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_36: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_96: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_36, [1, 128, 2560]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_97: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_96, [128, 2560]);  view_96 = None
    permute_50: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_25: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg72_1, view_97, permute_50);  arg72_1 = view_97 = permute_50 = None
    view_98: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 2560]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_33: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_29, view_98);  add_29 = view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_33, getitem_19);  getitem_19 = None
    add_34: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    mul_34: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_35: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_34, arg73_1);  mul_34 = arg73_1 = None
    add_35: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_35, arg74_1);  mul_35 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_99: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_35, [128, 2560]);  add_35 = None
    permute_51: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    addmm_26: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg76_1, view_99, permute_51);  arg76_1 = view_99 = permute_51 = None
    view_100: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 2560]);  addmm_26 = None
    mul_36: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_100, 0.11180339887498948);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_107: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_36, [1, 128, 32, 80]);  mul_36 = None
    permute_56: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    clone_40: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_108: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_40, [32, -1, 80]);  clone_40 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_66: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_108, 0);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_101: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_52: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_27: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg78_1, view_101, permute_52);  arg78_1 = view_101 = permute_52 = None
    view_102: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 2560]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_103: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_102, [1, -1, 32, 80]);  view_102 = None
    permute_53: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
    clone_38: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_109: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_38, [32, -1, 80]);  clone_38 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_67: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_109, 0);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_104: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_54: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_28: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg80_1, view_104, permute_54);  arg80_1 = view_104 = permute_54 = None
    view_105: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 2560]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_106: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_105, [1, -1, 32, 80]);  view_105 = None
    permute_55: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_39: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_110: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_39, [32, -1, 80]);  clone_39 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_68: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
    _scaled_dot_product_flash_attention_default_22 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_66, unsqueeze_default_67, unsqueeze_default_68, scale = 1.0);  unsqueeze_default_66 = unsqueeze_default_67 = unsqueeze_default_68 = None
    getitem_178: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_22[0];  _scaled_dot_product_flash_attention_default_22 = None
    squeeze_dim_22: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_178, 0);  getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_111: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_22, [1, 32, 128, 80]);  squeeze_dim_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_58: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_42: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    view_112: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 2560]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_113: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_112, [128, 2560]);  view_112 = None
    permute_59: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    addmm_29: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg82_1, view_113, permute_59);  arg82_1 = view_113 = permute_59 = None
    view_114: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 2560]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_36: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_33, view_114);  add_33 = view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_16: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_36, getitem_21);  getitem_21 = None
    add_37: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    mul_37: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_38: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_37, arg83_1);  mul_37 = arg83_1 = None
    add_38: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_38, arg84_1);  mul_38 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_115: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_38, [128, 2560]);  add_38 = None
    permute_60: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_30: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg86_1, view_115, permute_60);  arg86_1 = view_115 = permute_60 = None
    view_116: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 10240]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_40: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_3: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_39: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_41: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_39, add_39);  mul_39 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_117: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_41, [128, 10240]);  mul_41 = None
    permute_61: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_31: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg88_1, view_117, permute_61);  arg88_1 = view_117 = permute_61 = None
    view_118: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 2560]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_40: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_36, view_118);  add_36 = view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_40, getitem_23);  getitem_23 = None
    add_41: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_42: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_43: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_42, arg89_1);  mul_42 = arg89_1 = None
    add_42: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_43, arg90_1);  mul_43 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_119: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_42, [128, 2560])
    permute_62: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm_32: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg92_1, view_119, permute_62);  arg92_1 = view_119 = permute_62 = None
    view_120: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 2560]);  addmm_32 = None
    mul_44: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_120, 0.11180339887498948);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_44, [1, 128, 32, 80]);  mul_44 = None
    permute_67: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
    clone_48: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_128: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_48, [32, -1, 80]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_121: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_42, [128, 2560])
    permute_63: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_33: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg94_1, view_121, permute_63);  arg94_1 = view_121 = permute_63 = None
    view_122: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 2560]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_123: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_122, [1, -1, 32, 80]);  view_122 = None
    permute_64: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    clone_46: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_129: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_46, [32, -1, 80]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_68: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
    bmm_12: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_128, permute_68);  view_128 = permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_131: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [1, 32, 128, 128]);  bmm_12 = None
    add_43: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_131, expand_1);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_132: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_43, [32, 128, 128]);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_132, [-1], True)
    sub_18: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_132, amax_6);  view_132 = amax_6 = None
    exp_6: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_7: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_124: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_42, [128, 2560]);  add_42 = None
    permute_65: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    addmm_34: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg96_1, view_124, permute_65);  arg96_1 = view_124 = permute_65 = None
    view_125: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 2560]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_126: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_125, [1, -1, 32, 80]);  view_125 = None
    permute_66: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_47: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_130: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_47, [32, -1, 80]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_6, view_130);  div_6 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_133: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_13, [1, 32, 128, 80]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_69: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_50: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_134: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 2560]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_135: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_134, [128, 2560]);  view_134 = None
    permute_70: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    addmm_35: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg98_1, view_135, permute_70);  arg98_1 = view_135 = permute_70 = None
    view_136: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 2560]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_44: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_40, view_136);  add_40 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_19: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_44, getitem_25);  getitem_25 = None
    add_45: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    mul_45: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_46: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_45, arg99_1);  mul_45 = arg99_1 = None
    add_46: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_46, arg100_1);  mul_46 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_137: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_46, [128, 2560]);  add_46 = None
    permute_71: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    addmm_36: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg102_1, view_137, permute_71);  arg102_1 = view_137 = permute_71 = None
    view_138: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 2560]);  addmm_36 = None
    mul_47: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_138, 0.11180339887498948);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_47, [1, 128, 32, 80]);  mul_47 = None
    permute_76: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_54: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_146: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_54, [32, -1, 80]);  clone_54 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_63: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_146, 0);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_139: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_72: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    addmm_37: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg104_1, view_139, permute_72);  arg104_1 = view_139 = permute_72 = None
    view_140: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 2560]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_141: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_140, [1, -1, 32, 80]);  view_140 = None
    permute_73: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    clone_52: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_147: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_52, [32, -1, 80]);  clone_52 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_64: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_147, 0);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_142: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_74: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_38: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg106_1, view_142, permute_74);  arg106_1 = view_142 = permute_74 = None
    view_143: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 2560]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_144: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_143, [1, -1, 32, 80]);  view_143 = None
    permute_75: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
    clone_53: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_148: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_53, [32, -1, 80]);  clone_53 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_65: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_148, 0);  view_148 = None
    _scaled_dot_product_flash_attention_default_21 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_63, unsqueeze_default_64, unsqueeze_default_65, scale = 1.0);  unsqueeze_default_63 = unsqueeze_default_64 = unsqueeze_default_65 = None
    getitem_177: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_21[0];  _scaled_dot_product_flash_attention_default_21 = None
    squeeze_dim_21: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_177, 0);  getitem_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_149: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_21, [1, 32, 128, 80]);  squeeze_dim_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_78: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_56: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_150: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_56, [1, 128, 2560]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_151: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_150, [128, 2560]);  view_150 = None
    permute_79: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm_39: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg108_1, view_151, permute_79);  arg108_1 = view_151 = permute_79 = None
    view_152: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 2560]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_47: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_44, view_152);  add_44 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_21: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_47, getitem_27);  getitem_27 = None
    add_48: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    mul_48: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
    mul_49: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_48, arg109_1);  mul_48 = arg109_1 = None
    add_49: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_49, arg110_1);  mul_49 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_153: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_49, [128, 2560]);  add_49 = None
    permute_80: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    addmm_40: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg112_1, view_153, permute_80);  arg112_1 = view_153 = permute_80 = None
    view_154: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 10240]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_50: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
    mul_51: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_154, 0.7071067811865476);  view_154 = None
    erf_4: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_50: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_52: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_50, add_50);  mul_50 = add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_155: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_52, [128, 10240]);  mul_52 = None
    permute_81: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    addmm_41: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg114_1, view_155, permute_81);  arg114_1 = view_155 = permute_81 = None
    view_156: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 2560]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_51: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_47, view_156);  add_47 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_22: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_51, getitem_29);  getitem_29 = None
    add_52: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    mul_53: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_54: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_53, arg115_1);  mul_53 = arg115_1 = None
    add_53: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_54, arg116_1);  mul_54 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_157: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_53, [128, 2560])
    permute_82: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_42: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg118_1, view_157, permute_82);  arg118_1 = view_157 = permute_82 = None
    view_158: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 2560]);  addmm_42 = None
    mul_55: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_158, 0.11180339887498948);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_55, [1, 128, 32, 80]);  mul_55 = None
    permute_87: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    clone_62: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_166: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_62, [32, -1, 80]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_159: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_53, [128, 2560])
    permute_83: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_43: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg120_1, view_159, permute_83);  arg120_1 = view_159 = permute_83 = None
    view_160: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 2560]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_161: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_160, [1, -1, 32, 80]);  view_160 = None
    permute_84: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    clone_60: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_167: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_60, [32, -1, 80]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_88: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_16: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_166, permute_88);  view_166 = permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_169: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 32, 128, 128]);  bmm_16 = None
    add_54: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_169, expand_1);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_170: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_54, [32, 128, 128]);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_170, [-1], True)
    sub_23: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_170, amax_8);  view_170 = amax_8 = None
    exp_8: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_9: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_162: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_53, [128, 2560]);  add_53 = None
    permute_85: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_44: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg122_1, view_162, permute_85);  arg122_1 = view_162 = permute_85 = None
    view_163: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 2560]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_164: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_163, [1, -1, 32, 80]);  view_163 = None
    permute_86: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    clone_61: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_168: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_61, [32, -1, 80]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_8, view_168);  div_8 = view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_171: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_17, [1, 32, 128, 80]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_89: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_64: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
    view_172: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_64, [1, 128, 2560]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_173: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_172, [128, 2560]);  view_172 = None
    permute_90: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    addmm_45: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg124_1, view_173, permute_90);  arg124_1 = view_173 = permute_90 = None
    view_174: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 2560]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_55: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_51, view_174);  add_51 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_24: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_55, getitem_31);  getitem_31 = None
    add_56: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_56: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
    mul_57: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_56, arg125_1);  mul_56 = arg125_1 = None
    add_57: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_57, arg126_1);  mul_57 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_175: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_57, [128, 2560]);  add_57 = None
    permute_91: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_46: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg128_1, view_175, permute_91);  arg128_1 = view_175 = permute_91 = None
    view_176: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 2560]);  addmm_46 = None
    mul_58: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_176, 0.11180339887498948);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_183: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_58, [1, 128, 32, 80]);  mul_58 = None
    permute_96: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    clone_68: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_184: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_68, [32, -1, 80]);  clone_68 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_60: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_184, 0);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_177: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_92: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_47: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg130_1, view_177, permute_92);  arg130_1 = view_177 = permute_92 = None
    view_178: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 2560]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_179: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_178, [1, -1, 32, 80]);  view_178 = None
    permute_93: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
    clone_66: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_185: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_66, [32, -1, 80]);  clone_66 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_61: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_185, 0);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_180: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_94: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_48: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg132_1, view_180, permute_94);  arg132_1 = view_180 = permute_94 = None
    view_181: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 2560]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_182: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_181, [1, -1, 32, 80]);  view_181 = None
    permute_95: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_67: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_186: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_67, [32, -1, 80]);  clone_67 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_62: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_186, 0);  view_186 = None
    _scaled_dot_product_flash_attention_default_20 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_60, unsqueeze_default_61, unsqueeze_default_62, scale = 1.0);  unsqueeze_default_60 = unsqueeze_default_61 = unsqueeze_default_62 = None
    getitem_176: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_20[0];  _scaled_dot_product_flash_attention_default_20 = None
    squeeze_dim_20: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_176, 0);  getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_187: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_20, [1, 32, 128, 80]);  squeeze_dim_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_98: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_70: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    view_188: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_70, [1, 128, 2560]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_189: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_188, [128, 2560]);  view_188 = None
    permute_99: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_49: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg134_1, view_189, permute_99);  arg134_1 = view_189 = permute_99 = None
    view_190: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 2560]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_58: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_55, view_190);  add_55 = view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_26: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_58, getitem_33);  getitem_33 = None
    add_59: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    mul_59: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_16);  sub_26 = rsqrt_16 = None
    mul_60: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_59, arg135_1);  mul_59 = arg135_1 = None
    add_60: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_60, arg136_1);  mul_60 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_191: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_60, [128, 2560]);  add_60 = None
    permute_100: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_50: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg138_1, view_191, permute_100);  arg138_1 = view_191 = permute_100 = None
    view_192: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 10240]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_192, 0.5)
    mul_62: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476);  view_192 = None
    erf_5: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_61: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_63: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_61, add_61);  mul_61 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_193: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_63, [128, 10240]);  mul_63 = None
    permute_101: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_51: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg140_1, view_193, permute_101);  arg140_1 = view_193 = permute_101 = None
    view_194: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 2560]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_62: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_58, view_194);  add_58 = view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_27: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_62, getitem_35);  getitem_35 = None
    add_63: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    mul_64: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
    mul_65: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_64, arg141_1);  mul_64 = arg141_1 = None
    add_64: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_65, arg142_1);  mul_65 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_195: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_64, [128, 2560])
    permute_102: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_52: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg144_1, view_195, permute_102);  arg144_1 = view_195 = permute_102 = None
    view_196: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 2560]);  addmm_52 = None
    mul_66: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_196, 0.11180339887498948);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_203: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_66, [1, 128, 32, 80]);  mul_66 = None
    permute_107: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    clone_76: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_204: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_76, [32, -1, 80]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_197: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_64, [128, 2560])
    permute_103: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    addmm_53: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg146_1, view_197, permute_103);  arg146_1 = view_197 = permute_103 = None
    view_198: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 2560]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_199: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_198, [1, -1, 32, 80]);  view_198 = None
    permute_104: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
    clone_74: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_205: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_74, [32, -1, 80]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_108: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_20: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_204, permute_108);  view_204 = permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_207: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 32, 128, 128]);  bmm_20 = None
    add_65: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_207, expand_1);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_208: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_65, [32, 128, 128]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_208, [-1], True)
    sub_28: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_208, amax_10);  view_208 = amax_10 = None
    exp_10: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_11: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_200: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_64, [128, 2560]);  add_64 = None
    permute_105: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_54: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg148_1, view_200, permute_105);  arg148_1 = view_200 = permute_105 = None
    view_201: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 2560]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_202: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_201, [1, -1, 32, 80]);  view_201 = None
    permute_106: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    clone_75: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_206: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_75, [32, -1, 80]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_10, view_206);  div_10 = view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_209: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_21, [1, 32, 128, 80]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_109: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_78: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    view_210: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_78, [1, 128, 2560]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_211: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_210, [128, 2560]);  view_210 = None
    permute_110: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_55: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg150_1, view_211, permute_110);  arg150_1 = view_211 = permute_110 = None
    view_212: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 2560]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_66: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_62, view_212);  add_62 = view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_29: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_66, getitem_37);  getitem_37 = None
    add_67: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_67: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = rsqrt_18 = None
    mul_68: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_67, arg151_1);  mul_67 = arg151_1 = None
    add_68: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_68, arg152_1);  mul_68 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_213: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_68, [128, 2560]);  add_68 = None
    permute_111: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_56: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg154_1, view_213, permute_111);  arg154_1 = view_213 = permute_111 = None
    view_214: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 2560]);  addmm_56 = None
    mul_69: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_214, 0.11180339887498948);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_221: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_69, [1, 128, 32, 80]);  mul_69 = None
    permute_116: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
    clone_82: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_222: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_82, [32, -1, 80]);  clone_82 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_57: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_222, 0);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_215: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_112: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_57: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg156_1, view_215, permute_112);  arg156_1 = view_215 = permute_112 = None
    view_216: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_57, [1, 128, 2560]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_217: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_216, [1, -1, 32, 80]);  view_216 = None
    permute_113: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
    clone_80: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_223: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_80, [32, -1, 80]);  clone_80 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_58: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_223, 0);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_218: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_114: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_58: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg158_1, view_218, permute_114);  arg158_1 = view_218 = permute_114 = None
    view_219: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_58, [1, 128, 2560]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_220: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_219, [1, -1, 32, 80]);  view_219 = None
    permute_115: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
    clone_81: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_224: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_81, [32, -1, 80]);  clone_81 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_59: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_224, 0);  view_224 = None
    _scaled_dot_product_flash_attention_default_19 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_57, unsqueeze_default_58, unsqueeze_default_59, scale = 1.0);  unsqueeze_default_57 = unsqueeze_default_58 = unsqueeze_default_59 = None
    getitem_175: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_19[0];  _scaled_dot_product_flash_attention_default_19 = None
    squeeze_dim_19: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_175, 0);  getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_225: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_19, [1, 32, 128, 80]);  squeeze_dim_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_118: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_84: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_226: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_84, [1, 128, 2560]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_227: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_226, [128, 2560]);  view_226 = None
    permute_119: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_59: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg160_1, view_227, permute_119);  arg160_1 = view_227 = permute_119 = None
    view_228: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_59, [1, 128, 2560]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_69: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_66, view_228);  add_66 = view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_31: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_69, getitem_39);  getitem_39 = None
    add_70: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    mul_70: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_19);  sub_31 = rsqrt_19 = None
    mul_71: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_70, arg161_1);  mul_70 = arg161_1 = None
    add_71: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_71, arg162_1);  mul_71 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_229: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_71, [128, 2560]);  add_71 = None
    permute_120: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_60: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg164_1, view_229, permute_120);  arg164_1 = view_229 = permute_120 = None
    view_230: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_60, [1, 128, 10240]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_72: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_230, 0.5)
    mul_73: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_230, 0.7071067811865476);  view_230 = None
    erf_6: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
    add_72: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_74: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_72, add_72);  mul_72 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_231: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_74, [128, 10240]);  mul_74 = None
    permute_121: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_61: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg166_1, view_231, permute_121);  arg166_1 = view_231 = permute_121 = None
    view_232: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_61, [1, 128, 2560]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_73: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_69, view_232);  add_69 = view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_32: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_73, getitem_41);  getitem_41 = None
    add_74: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_75: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = rsqrt_20 = None
    mul_76: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_75, arg167_1);  mul_75 = arg167_1 = None
    add_75: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_76, arg168_1);  mul_76 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_233: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_75, [128, 2560])
    permute_122: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_62: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg170_1, view_233, permute_122);  arg170_1 = view_233 = permute_122 = None
    view_234: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_62, [1, 128, 2560]);  addmm_62 = None
    mul_77: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_234, 0.11180339887498948);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_241: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_77, [1, 128, 32, 80]);  mul_77 = None
    permute_127: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    clone_90: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_242: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_90, [32, -1, 80]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_235: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_75, [128, 2560])
    permute_123: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_63: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg172_1, view_235, permute_123);  arg172_1 = view_235 = permute_123 = None
    view_236: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_63, [1, 128, 2560]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_237: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_236, [1, -1, 32, 80]);  view_236 = None
    permute_124: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
    clone_88: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_243: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_88, [32, -1, 80]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_128: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    bmm_24: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_242, permute_128);  view_242 = permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_245: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 32, 128, 128]);  bmm_24 = None
    add_76: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_245, expand_1);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_246: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_76, [32, 128, 128]);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_246, [-1], True)
    sub_33: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_246, amax_12);  view_246 = amax_12 = None
    exp_12: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_13: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_238: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_75, [128, 2560]);  add_75 = None
    permute_125: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_64: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg174_1, view_238, permute_125);  arg174_1 = view_238 = permute_125 = None
    view_239: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_64, [1, 128, 2560]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_240: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_239, [1, -1, 32, 80]);  view_239 = None
    permute_126: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    clone_89: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_244: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_89, [32, -1, 80]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_12, view_244);  div_12 = view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_247: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_25, [1, 32, 128, 80]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_129: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_92: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_248: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_92, [1, 128, 2560]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_249: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_248, [128, 2560]);  view_248 = None
    permute_130: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_65: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg176_1, view_249, permute_130);  arg176_1 = view_249 = permute_130 = None
    view_250: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_65, [1, 128, 2560]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_77: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_73, view_250);  add_73 = view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_34: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_77, getitem_43);  getitem_43 = None
    add_78: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_78: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = rsqrt_21 = None
    mul_79: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_78, arg177_1);  mul_78 = arg177_1 = None
    add_79: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_79, arg178_1);  mul_79 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_251: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_79, [128, 2560]);  add_79 = None
    permute_131: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_66: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg180_1, view_251, permute_131);  arg180_1 = view_251 = permute_131 = None
    view_252: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_66, [1, 128, 2560]);  addmm_66 = None
    mul_80: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_252, 0.11180339887498948);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_259: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_80, [1, 128, 32, 80]);  mul_80 = None
    permute_136: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    clone_96: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_260: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_96, [32, -1, 80]);  clone_96 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_54: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_260, 0);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_253: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_132: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_67: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg182_1, view_253, permute_132);  arg182_1 = view_253 = permute_132 = None
    view_254: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_67, [1, 128, 2560]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_255: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_254, [1, -1, 32, 80]);  view_254 = None
    permute_133: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    clone_94: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_261: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_94, [32, -1, 80]);  clone_94 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_55: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_261, 0);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_256: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_134: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_68: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg184_1, view_256, permute_134);  arg184_1 = view_256 = permute_134 = None
    view_257: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_68, [1, 128, 2560]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_258: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_257, [1, -1, 32, 80]);  view_257 = None
    permute_135: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_95: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_262: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_95, [32, -1, 80]);  clone_95 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_56: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_262, 0);  view_262 = None
    _scaled_dot_product_flash_attention_default_18 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_54, unsqueeze_default_55, unsqueeze_default_56, scale = 1.0);  unsqueeze_default_54 = unsqueeze_default_55 = unsqueeze_default_56 = None
    getitem_174: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_18[0];  _scaled_dot_product_flash_attention_default_18 = None
    squeeze_dim_18: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_174, 0);  getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_263: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_18, [1, 32, 128, 80]);  squeeze_dim_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_138: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_98: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
    view_264: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_98, [1, 128, 2560]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_265: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_264, [128, 2560]);  view_264 = None
    permute_139: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_69: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg186_1, view_265, permute_139);  arg186_1 = view_265 = permute_139 = None
    view_266: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_69, [1, 128, 2560]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_80: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_77, view_266);  add_77 = view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_36: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_80, getitem_45);  getitem_45 = None
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_81: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_22);  sub_36 = rsqrt_22 = None
    mul_82: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_81, arg187_1);  mul_81 = arg187_1 = None
    add_82: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_82, arg188_1);  mul_82 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_267: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_82, [128, 2560]);  add_82 = None
    permute_140: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_70: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg190_1, view_267, permute_140);  arg190_1 = view_267 = permute_140 = None
    view_268: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_70, [1, 128, 10240]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_268, 0.5)
    mul_84: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_268, 0.7071067811865476);  view_268 = None
    erf_7: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
    add_83: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_85: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_83, add_83);  mul_83 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_269: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_85, [128, 10240]);  mul_85 = None
    permute_141: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    addmm_71: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg192_1, view_269, permute_141);  arg192_1 = view_269 = permute_141 = None
    view_270: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_71, [1, 128, 2560]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_84: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_80, view_270);  add_80 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_37: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_84, getitem_47);  getitem_47 = None
    add_85: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_86: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = rsqrt_23 = None
    mul_87: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_86, arg193_1);  mul_86 = arg193_1 = None
    add_86: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_87, arg194_1);  mul_87 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_271: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_86, [128, 2560])
    permute_142: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_72: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg196_1, view_271, permute_142);  arg196_1 = view_271 = permute_142 = None
    view_272: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_72, [1, 128, 2560]);  addmm_72 = None
    mul_88: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_272, 0.11180339887498948);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_279: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_88, [1, 128, 32, 80]);  mul_88 = None
    permute_147: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    clone_104: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_280: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_104, [32, -1, 80]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_273: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_86, [128, 2560])
    permute_143: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_73: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg198_1, view_273, permute_143);  arg198_1 = view_273 = permute_143 = None
    view_274: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_73, [1, 128, 2560]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_275: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_274, [1, -1, 32, 80]);  view_274 = None
    permute_144: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
    clone_102: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_281: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_102, [32, -1, 80]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_148: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
    bmm_28: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_280, permute_148);  view_280 = permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_283: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 32, 128, 128]);  bmm_28 = None
    add_87: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_283, expand_1);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_284: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_87, [32, 128, 128]);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_284, [-1], True)
    sub_38: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_284, amax_14);  view_284 = amax_14 = None
    exp_14: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_15: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_276: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_86, [128, 2560]);  add_86 = None
    permute_145: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    addmm_74: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg200_1, view_276, permute_145);  arg200_1 = view_276 = permute_145 = None
    view_277: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_74, [1, 128, 2560]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_278: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_277, [1, -1, 32, 80]);  view_277 = None
    permute_146: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    clone_103: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_282: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_103, [32, -1, 80]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_14, view_282);  div_14 = view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_285: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_29, [1, 32, 128, 80]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_149: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_106: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
    view_286: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_106, [1, 128, 2560]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_287: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_286, [128, 2560]);  view_286 = None
    permute_150: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    addmm_75: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg202_1, view_287, permute_150);  arg202_1 = view_287 = permute_150 = None
    view_288: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_75, [1, 128, 2560]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_88: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_84, view_288);  add_84 = view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_39: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_88, getitem_49);  getitem_49 = None
    add_89: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_89: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = rsqrt_24 = None
    mul_90: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_89, arg203_1);  mul_89 = arg203_1 = None
    add_90: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_90, arg204_1);  mul_90 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_289: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_90, [128, 2560]);  add_90 = None
    permute_151: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    addmm_76: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg206_1, view_289, permute_151);  arg206_1 = view_289 = permute_151 = None
    view_290: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_76, [1, 128, 2560]);  addmm_76 = None
    mul_91: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_290, 0.11180339887498948);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_297: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_91, [1, 128, 32, 80]);  mul_91 = None
    permute_156: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
    clone_110: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_298: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_110, [32, -1, 80]);  clone_110 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_51: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_298, 0);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_291: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_152: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
    addmm_77: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg208_1, view_291, permute_152);  arg208_1 = view_291 = permute_152 = None
    view_292: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_77, [1, 128, 2560]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_293: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_292, [1, -1, 32, 80]);  view_292 = None
    permute_153: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    clone_108: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_299: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_108, [32, -1, 80]);  clone_108 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_52: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_299, 0);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_294: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_154: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
    addmm_78: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg210_1, view_294, permute_154);  arg210_1 = view_294 = permute_154 = None
    view_295: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_78, [1, 128, 2560]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_296: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_295, [1, -1, 32, 80]);  view_295 = None
    permute_155: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    clone_109: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_300: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_109, [32, -1, 80]);  clone_109 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_53: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_300, 0);  view_300 = None
    _scaled_dot_product_flash_attention_default_17 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_51, unsqueeze_default_52, unsqueeze_default_53, scale = 1.0);  unsqueeze_default_51 = unsqueeze_default_52 = unsqueeze_default_53 = None
    getitem_173: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_17[0];  _scaled_dot_product_flash_attention_default_17 = None
    squeeze_dim_17: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_173, 0);  getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_301: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_17, [1, 32, 128, 80]);  squeeze_dim_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_158: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_112: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    view_302: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_112, [1, 128, 2560]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_303: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_302, [128, 2560]);  view_302 = None
    permute_159: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
    addmm_79: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg212_1, view_303, permute_159);  arg212_1 = view_303 = permute_159 = None
    view_304: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_79, [1, 128, 2560]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_91: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_88, view_304);  add_88 = view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_41: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_91, getitem_51);  getitem_51 = None
    add_92: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    mul_92: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_25);  sub_41 = rsqrt_25 = None
    mul_93: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_92, arg213_1);  mul_92 = arg213_1 = None
    add_93: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_93, arg214_1);  mul_93 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_305: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_93, [128, 2560]);  add_93 = None
    permute_160: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    addmm_80: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg216_1, view_305, permute_160);  arg216_1 = view_305 = permute_160 = None
    view_306: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_80, [1, 128, 10240]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_306, 0.5)
    mul_95: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_306, 0.7071067811865476);  view_306 = None
    erf_8: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_94: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_96: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_94, add_94);  mul_94 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_307: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_96, [128, 10240]);  mul_96 = None
    permute_161: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    addmm_81: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg218_1, view_307, permute_161);  arg218_1 = view_307 = permute_161 = None
    view_308: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_81, [1, 128, 2560]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_95: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_91, view_308);  add_91 = view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    sub_42: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_95, getitem_53);  getitem_53 = None
    add_96: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    mul_97: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_26);  sub_42 = rsqrt_26 = None
    mul_98: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_97, arg219_1);  mul_97 = arg219_1 = None
    add_97: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_98, arg220_1);  mul_98 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_309: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_97, [128, 2560])
    permute_162: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
    addmm_82: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg222_1, view_309, permute_162);  arg222_1 = view_309 = permute_162 = None
    view_310: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_82, [1, 128, 2560]);  addmm_82 = None
    mul_99: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_310, 0.11180339887498948);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_317: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_99, [1, 128, 32, 80]);  mul_99 = None
    permute_167: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    clone_118: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_318: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_118, [32, -1, 80]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_311: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_97, [128, 2560])
    permute_163: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
    addmm_83: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg224_1, view_311, permute_163);  arg224_1 = view_311 = permute_163 = None
    view_312: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_83, [1, 128, 2560]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_313: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_312, [1, -1, 32, 80]);  view_312 = None
    permute_164: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    clone_116: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_319: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_116, [32, -1, 80]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_168: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None
    bmm_32: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_318, permute_168);  view_318 = permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_321: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 32, 128, 128]);  bmm_32 = None
    add_98: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_321, expand_1);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_322: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_98, [32, 128, 128]);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_322, [-1], True)
    sub_43: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_322, amax_16);  view_322 = amax_16 = None
    exp_16: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_17: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_314: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_97, [128, 2560]);  add_97 = None
    permute_165: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    addmm_84: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg226_1, view_314, permute_165);  arg226_1 = view_314 = permute_165 = None
    view_315: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_84, [1, 128, 2560]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_316: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_315, [1, -1, 32, 80]);  view_315 = None
    permute_166: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    clone_117: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_320: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_117, [32, -1, 80]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_16, view_320);  div_16 = view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_323: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_33, [1, 32, 128, 80]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_169: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_120: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    view_324: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_120, [1, 128, 2560]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_325: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_324, [128, 2560]);  view_324 = None
    permute_170: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    addmm_85: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg228_1, view_325, permute_170);  arg228_1 = view_325 = permute_170 = None
    view_326: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_85, [1, 128, 2560]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_99: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_95, view_326);  add_95 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_44: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_99, getitem_55);  getitem_55 = None
    add_100: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_100: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_27);  sub_44 = rsqrt_27 = None
    mul_101: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_100, arg229_1);  mul_100 = arg229_1 = None
    add_101: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_101, arg230_1);  mul_101 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_327: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_101, [128, 2560]);  add_101 = None
    permute_171: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
    addmm_86: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg232_1, view_327, permute_171);  arg232_1 = view_327 = permute_171 = None
    view_328: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_86, [1, 128, 2560]);  addmm_86 = None
    mul_102: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_328, 0.11180339887498948);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_335: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_102, [1, 128, 32, 80]);  mul_102 = None
    permute_176: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
    clone_124: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_336: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_124, [32, -1, 80]);  clone_124 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_48: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_336, 0);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_329: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_172: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
    addmm_87: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg234_1, view_329, permute_172);  arg234_1 = view_329 = permute_172 = None
    view_330: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_87, [1, 128, 2560]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_331: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_330, [1, -1, 32, 80]);  view_330 = None
    permute_173: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    clone_122: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_337: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_122, [32, -1, 80]);  clone_122 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_49: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_337, 0);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_332: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_174: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
    addmm_88: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg236_1, view_332, permute_174);  arg236_1 = view_332 = permute_174 = None
    view_333: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_88, [1, 128, 2560]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_334: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_333, [1, -1, 32, 80]);  view_333 = None
    permute_175: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    clone_123: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_338: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_123, [32, -1, 80]);  clone_123 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_50: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_338, 0);  view_338 = None
    _scaled_dot_product_flash_attention_default_16 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_48, unsqueeze_default_49, unsqueeze_default_50, scale = 1.0);  unsqueeze_default_48 = unsqueeze_default_49 = unsqueeze_default_50 = None
    getitem_172: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_16[0];  _scaled_dot_product_flash_attention_default_16 = None
    squeeze_dim_16: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_172, 0);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_339: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_16, [1, 32, 128, 80]);  squeeze_dim_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_178: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_126: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
    view_340: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_126, [1, 128, 2560]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_341: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_340, [128, 2560]);  view_340 = None
    permute_179: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    addmm_89: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg238_1, view_341, permute_179);  arg238_1 = view_341 = permute_179 = None
    view_342: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_89, [1, 128, 2560]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_102: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_99, view_342);  add_99 = view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_46: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_102, getitem_57);  getitem_57 = None
    add_103: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    mul_103: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_28);  sub_46 = rsqrt_28 = None
    mul_104: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_103, arg239_1);  mul_103 = arg239_1 = None
    add_104: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_104, arg240_1);  mul_104 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_343: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_104, [128, 2560]);  add_104 = None
    permute_180: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
    addmm_90: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg242_1, view_343, permute_180);  arg242_1 = view_343 = permute_180 = None
    view_344: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_90, [1, 128, 10240]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_105: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_344, 0.5)
    mul_106: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_344, 0.7071067811865476);  view_344 = None
    erf_9: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
    add_105: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_107: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_105, add_105);  mul_105 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_345: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_107, [128, 10240]);  mul_107 = None
    permute_181: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    addmm_91: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg244_1, view_345, permute_181);  arg244_1 = view_345 = permute_181 = None
    view_346: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_91, [1, 128, 2560]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_106: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_102, view_346);  add_102 = view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_47: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_106, getitem_59);  getitem_59 = None
    add_107: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    mul_108: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_29);  sub_47 = rsqrt_29 = None
    mul_109: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_108, arg245_1);  mul_108 = arg245_1 = None
    add_108: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_109, arg246_1);  mul_109 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_347: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_108, [128, 2560])
    permute_182: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    addmm_92: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg248_1, view_347, permute_182);  arg248_1 = view_347 = permute_182 = None
    view_348: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_92, [1, 128, 2560]);  addmm_92 = None
    mul_110: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_348, 0.11180339887498948);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_355: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_110, [1, 128, 32, 80]);  mul_110 = None
    permute_187: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_355, [0, 2, 1, 3]);  view_355 = None
    clone_132: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_356: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_132, [32, -1, 80]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_349: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_108, [128, 2560])
    permute_183: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    addmm_93: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg250_1, view_349, permute_183);  arg250_1 = view_349 = permute_183 = None
    view_350: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_93, [1, 128, 2560]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_351: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_350, [1, -1, 32, 80]);  view_350 = None
    permute_184: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
    clone_130: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_357: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_130, [32, -1, 80]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_188: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
    bmm_36: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_356, permute_188);  view_356 = permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_359: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 32, 128, 128]);  bmm_36 = None
    add_109: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_359, expand_1);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_360: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_109, [32, 128, 128]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_360, [-1], True)
    sub_48: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_360, amax_18);  view_360 = amax_18 = None
    exp_18: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_19: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_352: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_108, [128, 2560]);  add_108 = None
    permute_185: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
    addmm_94: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg252_1, view_352, permute_185);  arg252_1 = view_352 = permute_185 = None
    view_353: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_94, [1, 128, 2560]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_354: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_353, [1, -1, 32, 80]);  view_353 = None
    permute_186: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
    clone_131: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_358: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_131, [32, -1, 80]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_37: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_18, view_358);  div_18 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_361: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_37, [1, 32, 128, 80]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_189: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_134: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_362: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_134, [1, 128, 2560]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_363: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_362, [128, 2560]);  view_362 = None
    permute_190: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    addmm_95: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg254_1, view_363, permute_190);  arg254_1 = view_363 = permute_190 = None
    view_364: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_95, [1, 128, 2560]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_110: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_106, view_364);  add_106 = view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_49: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_110, getitem_61);  getitem_61 = None
    add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_111: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_30);  sub_49 = rsqrt_30 = None
    mul_112: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_111, arg255_1);  mul_111 = arg255_1 = None
    add_112: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_112, arg256_1);  mul_112 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_365: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_112, [128, 2560]);  add_112 = None
    permute_191: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
    addmm_96: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg258_1, view_365, permute_191);  arg258_1 = view_365 = permute_191 = None
    view_366: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_96, [1, 128, 2560]);  addmm_96 = None
    mul_113: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_366, 0.11180339887498948);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_373: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_113, [1, 128, 32, 80]);  mul_113 = None
    permute_196: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    clone_138: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_374: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_138, [32, -1, 80]);  clone_138 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_45: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_374, 0);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_367: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_192: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    addmm_97: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg260_1, view_367, permute_192);  arg260_1 = view_367 = permute_192 = None
    view_368: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_97, [1, 128, 2560]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_369: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_368, [1, -1, 32, 80]);  view_368 = None
    permute_193: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
    clone_136: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_375: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_136, [32, -1, 80]);  clone_136 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_46: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_375, 0);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_370: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_194: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    addmm_98: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg262_1, view_370, permute_194);  arg262_1 = view_370 = permute_194 = None
    view_371: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_98, [1, 128, 2560]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_372: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_371, [1, -1, 32, 80]);  view_371 = None
    permute_195: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    clone_137: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_376: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_137, [32, -1, 80]);  clone_137 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_47: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_376, 0);  view_376 = None
    _scaled_dot_product_flash_attention_default_15 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_45, unsqueeze_default_46, unsqueeze_default_47, scale = 1.0);  unsqueeze_default_45 = unsqueeze_default_46 = unsqueeze_default_47 = None
    getitem_171: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_15[0];  _scaled_dot_product_flash_attention_default_15 = None
    squeeze_dim_15: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_171, 0);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_377: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_15, [1, 32, 128, 80]);  squeeze_dim_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_198: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_140: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    view_378: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_140, [1, 128, 2560]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_379: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_378, [128, 2560]);  view_378 = None
    permute_199: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
    addmm_99: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg264_1, view_379, permute_199);  arg264_1 = view_379 = permute_199 = None
    view_380: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_99, [1, 128, 2560]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_113: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_110, view_380);  add_110 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    sub_51: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_113, getitem_63);  getitem_63 = None
    add_114: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    mul_114: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_31);  sub_51 = rsqrt_31 = None
    mul_115: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_114, arg265_1);  mul_114 = arg265_1 = None
    add_115: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_115, arg266_1);  mul_115 = arg266_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_381: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_115, [128, 2560]);  add_115 = None
    permute_200: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    addmm_100: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg268_1, view_381, permute_200);  arg268_1 = view_381 = permute_200 = None
    view_382: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_100, [1, 128, 10240]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_116: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_382, 0.5)
    mul_117: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_382, 0.7071067811865476);  view_382 = None
    erf_10: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_116: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_118: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_116, add_116);  mul_116 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_383: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_118, [128, 10240]);  mul_118 = None
    permute_201: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
    addmm_101: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg270_1, view_383, permute_201);  arg270_1 = view_383 = permute_201 = None
    view_384: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_101, [1, 128, 2560]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_117: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_113, view_384);  add_113 = view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    sub_52: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_117, getitem_65);  getitem_65 = None
    add_118: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    mul_119: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_32);  sub_52 = rsqrt_32 = None
    mul_120: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_119, arg271_1);  mul_119 = arg271_1 = None
    add_119: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_120, arg272_1);  mul_120 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_385: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_119, [128, 2560])
    permute_202: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
    addmm_102: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg274_1, view_385, permute_202);  arg274_1 = view_385 = permute_202 = None
    view_386: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_102, [1, 128, 2560]);  addmm_102 = None
    mul_121: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_386, 0.11180339887498948);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_393: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_121, [1, 128, 32, 80]);  mul_121 = None
    permute_207: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    clone_146: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_394: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_146, [32, -1, 80]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_387: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_119, [128, 2560])
    permute_203: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
    addmm_103: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg276_1, view_387, permute_203);  arg276_1 = view_387 = permute_203 = None
    view_388: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_103, [1, 128, 2560]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_389: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_388, [1, -1, 32, 80]);  view_388 = None
    permute_204: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None
    clone_144: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_395: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_144, [32, -1, 80]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_208: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_395, [0, 2, 1]);  view_395 = None
    bmm_40: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_394, permute_208);  view_394 = permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_397: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 32, 128, 128]);  bmm_40 = None
    add_120: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_397, expand_1);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_398: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_120, [32, 128, 128]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_398, [-1], True)
    sub_53: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_398, amax_20);  view_398 = amax_20 = None
    exp_20: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
    sum_21: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_390: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_119, [128, 2560]);  add_119 = None
    permute_205: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    addmm_104: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg278_1, view_390, permute_205);  arg278_1 = view_390 = permute_205 = None
    view_391: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_104, [1, 128, 2560]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_392: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_391, [1, -1, 32, 80]);  view_391 = None
    permute_206: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    clone_145: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_396: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_145, [32, -1, 80]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_41: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_20, view_396);  div_20 = view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_399: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_41, [1, 32, 128, 80]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_209: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_148: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_400: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_148, [1, 128, 2560]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_401: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_400, [128, 2560]);  view_400 = None
    permute_210: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
    addmm_105: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg280_1, view_401, permute_210);  arg280_1 = view_401 = permute_210 = None
    view_402: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_105, [1, 128, 2560]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_121: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_117, view_402);  add_117 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    sub_54: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_121, getitem_67);  getitem_67 = None
    add_122: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    mul_122: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_33);  sub_54 = rsqrt_33 = None
    mul_123: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_122, arg281_1);  mul_122 = arg281_1 = None
    add_123: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_123, arg282_1);  mul_123 = arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_403: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_123, [128, 2560]);  add_123 = None
    permute_211: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
    addmm_106: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg284_1, view_403, permute_211);  arg284_1 = view_403 = permute_211 = None
    view_404: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_106, [1, 128, 2560]);  addmm_106 = None
    mul_124: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_404, 0.11180339887498948);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_411: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_124, [1, 128, 32, 80]);  mul_124 = None
    permute_216: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
    clone_152: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_412: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_152, [32, -1, 80]);  clone_152 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_42: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_412, 0);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_405: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_212: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
    addmm_107: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg286_1, view_405, permute_212);  arg286_1 = view_405 = permute_212 = None
    view_406: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_107, [1, 128, 2560]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_407: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_406, [1, -1, 32, 80]);  view_406 = None
    permute_213: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
    clone_150: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_413: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_150, [32, -1, 80]);  clone_150 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_43: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_413, 0);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_408: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_214: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
    addmm_108: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg288_1, view_408, permute_214);  arg288_1 = view_408 = permute_214 = None
    view_409: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_108, [1, 128, 2560]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_410: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_409, [1, -1, 32, 80]);  view_409 = None
    permute_215: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    clone_151: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_414: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_151, [32, -1, 80]);  clone_151 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_44: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_414, 0);  view_414 = None
    _scaled_dot_product_flash_attention_default_14 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_42, unsqueeze_default_43, unsqueeze_default_44, scale = 1.0);  unsqueeze_default_42 = unsqueeze_default_43 = unsqueeze_default_44 = None
    getitem_170: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_14[0];  _scaled_dot_product_flash_attention_default_14 = None
    squeeze_dim_14: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_170, 0);  getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_415: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_14, [1, 32, 128, 80]);  squeeze_dim_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_218: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_154: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_416: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_154, [1, 128, 2560]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_417: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_416, [128, 2560]);  view_416 = None
    permute_219: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
    addmm_109: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg290_1, view_417, permute_219);  arg290_1 = view_417 = permute_219 = None
    view_418: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_109, [1, 128, 2560]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_124: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_121, view_418);  add_121 = view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_124, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    sub_56: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_124, getitem_69);  getitem_69 = None
    add_125: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    mul_125: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_34);  sub_56 = rsqrt_34 = None
    mul_126: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_125, arg291_1);  mul_125 = arg291_1 = None
    add_126: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_126, arg292_1);  mul_126 = arg292_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_419: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_126, [128, 2560]);  add_126 = None
    permute_220: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    addmm_110: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg294_1, view_419, permute_220);  arg294_1 = view_419 = permute_220 = None
    view_420: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_110, [1, 128, 10240]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_127: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_420, 0.5)
    mul_128: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_420, 0.7071067811865476);  view_420 = None
    erf_11: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_127: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_129: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_127, add_127);  mul_127 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_421: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_129, [128, 10240]);  mul_129 = None
    permute_221: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
    addmm_111: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg296_1, view_421, permute_221);  arg296_1 = view_421 = permute_221 = None
    view_422: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_111, [1, 128, 2560]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_128: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_124, view_422);  add_124 = view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    sub_57: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_128, getitem_71);  getitem_71 = None
    add_129: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    mul_130: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_35);  sub_57 = rsqrt_35 = None
    mul_131: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_130, arg297_1);  mul_130 = arg297_1 = None
    add_130: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_131, arg298_1);  mul_131 = arg298_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_423: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_130, [128, 2560])
    permute_222: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
    addmm_112: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg300_1, view_423, permute_222);  arg300_1 = view_423 = permute_222 = None
    view_424: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_112, [1, 128, 2560]);  addmm_112 = None
    mul_132: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_424, 0.11180339887498948);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_431: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_132, [1, 128, 32, 80]);  mul_132 = None
    permute_227: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
    clone_160: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_432: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_160, [32, -1, 80]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_425: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_130, [128, 2560])
    permute_223: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
    addmm_113: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg302_1, view_425, permute_223);  arg302_1 = view_425 = permute_223 = None
    view_426: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_113, [1, 128, 2560]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_427: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_426, [1, -1, 32, 80]);  view_426 = None
    permute_224: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    clone_158: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_433: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_158, [32, -1, 80]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_228: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_433, [0, 2, 1]);  view_433 = None
    bmm_44: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_432, permute_228);  view_432 = permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_435: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 32, 128, 128]);  bmm_44 = None
    add_131: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_435, expand_1);  view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_436: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_131, [32, 128, 128]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_436, [-1], True)
    sub_58: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_436, amax_22);  view_436 = amax_22 = None
    exp_22: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_23: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_428: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_130, [128, 2560]);  add_130 = None
    permute_225: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
    addmm_114: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg304_1, view_428, permute_225);  arg304_1 = view_428 = permute_225 = None
    view_429: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_114, [1, 128, 2560]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_430: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_429, [1, -1, 32, 80]);  view_429 = None
    permute_226: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    clone_159: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_434: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_159, [32, -1, 80]);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_45: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_22, view_434);  div_22 = view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_437: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_45, [1, 32, 128, 80]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_229: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_162: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
    view_438: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_162, [1, 128, 2560]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_439: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_438, [128, 2560]);  view_438 = None
    permute_230: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
    addmm_115: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg306_1, view_439, permute_230);  arg306_1 = view_439 = permute_230 = None
    view_440: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_115, [1, 128, 2560]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_132: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_128, view_440);  add_128 = view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    sub_59: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_132, getitem_73);  getitem_73 = None
    add_133: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    mul_133: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_36);  sub_59 = rsqrt_36 = None
    mul_134: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_133, arg307_1);  mul_133 = arg307_1 = None
    add_134: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_134, arg308_1);  mul_134 = arg308_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_441: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_134, [128, 2560]);  add_134 = None
    permute_231: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
    addmm_116: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg310_1, view_441, permute_231);  arg310_1 = view_441 = permute_231 = None
    view_442: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_116, [1, 128, 2560]);  addmm_116 = None
    mul_135: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_442, 0.11180339887498948);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_449: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_135, [1, 128, 32, 80]);  mul_135 = None
    permute_236: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
    clone_166: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_450: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_166, [32, -1, 80]);  clone_166 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_39: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_450, 0);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_443: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_232: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
    addmm_117: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg312_1, view_443, permute_232);  arg312_1 = view_443 = permute_232 = None
    view_444: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_117, [1, 128, 2560]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_445: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_444, [1, -1, 32, 80]);  view_444 = None
    permute_233: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    clone_164: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_451: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_164, [32, -1, 80]);  clone_164 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_40: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_451, 0);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_446: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_234: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
    addmm_118: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg314_1, view_446, permute_234);  arg314_1 = view_446 = permute_234 = None
    view_447: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_118, [1, 128, 2560]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_448: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_447, [1, -1, 32, 80]);  view_447 = None
    permute_235: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    clone_165: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_452: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_165, [32, -1, 80]);  clone_165 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_41: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_452, 0);  view_452 = None
    _scaled_dot_product_flash_attention_default_13 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_39, unsqueeze_default_40, unsqueeze_default_41, scale = 1.0);  unsqueeze_default_39 = unsqueeze_default_40 = unsqueeze_default_41 = None
    getitem_169: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_13[0];  _scaled_dot_product_flash_attention_default_13 = None
    squeeze_dim_13: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_169, 0);  getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_453: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_13, [1, 32, 128, 80]);  squeeze_dim_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_238: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_168: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    view_454: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_168, [1, 128, 2560]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_455: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_454, [128, 2560]);  view_454 = None
    permute_239: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
    addmm_119: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg316_1, view_455, permute_239);  arg316_1 = view_455 = permute_239 = None
    view_456: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_119, [1, 128, 2560]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_135: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_132, view_456);  add_132 = view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    sub_61: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_135, getitem_75);  getitem_75 = None
    add_136: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    mul_136: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_37);  sub_61 = rsqrt_37 = None
    mul_137: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_136, arg317_1);  mul_136 = arg317_1 = None
    add_137: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_137, arg318_1);  mul_137 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_457: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_137, [128, 2560]);  add_137 = None
    permute_240: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
    addmm_120: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg320_1, view_457, permute_240);  arg320_1 = view_457 = permute_240 = None
    view_458: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_120, [1, 128, 10240]);  addmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_138: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_458, 0.5)
    mul_139: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_458, 0.7071067811865476);  view_458 = None
    erf_12: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_138: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_140: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_138, add_138);  mul_138 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_459: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_140, [128, 10240]);  mul_140 = None
    permute_241: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
    addmm_121: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg322_1, view_459, permute_241);  arg322_1 = view_459 = permute_241 = None
    view_460: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_121, [1, 128, 2560]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_139: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_135, view_460);  add_135 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    sub_62: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_139, getitem_77);  getitem_77 = None
    add_140: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    mul_141: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_38);  sub_62 = rsqrt_38 = None
    mul_142: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_141, arg323_1);  mul_141 = arg323_1 = None
    add_141: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_142, arg324_1);  mul_142 = arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_461: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_141, [128, 2560])
    permute_242: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
    addmm_122: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg326_1, view_461, permute_242);  arg326_1 = view_461 = permute_242 = None
    view_462: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_122, [1, 128, 2560]);  addmm_122 = None
    mul_143: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_462, 0.11180339887498948);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_469: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_143, [1, 128, 32, 80]);  mul_143 = None
    permute_247: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
    clone_174: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_470: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_174, [32, -1, 80]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_463: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_141, [128, 2560])
    permute_243: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
    addmm_123: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg328_1, view_463, permute_243);  arg328_1 = view_463 = permute_243 = None
    view_464: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_123, [1, 128, 2560]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_465: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_464, [1, -1, 32, 80]);  view_464 = None
    permute_244: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
    clone_172: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_471: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_172, [32, -1, 80]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_248: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
    bmm_48: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_470, permute_248);  view_470 = permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_473: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_48, [1, 32, 128, 128]);  bmm_48 = None
    add_142: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_473, expand_1);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_474: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_142, [32, 128, 128]);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_24: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_474, [-1], True)
    sub_63: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_474, amax_24);  view_474 = amax_24 = None
    exp_24: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_25: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_24: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_466: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_141, [128, 2560]);  add_141 = None
    permute_245: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
    addmm_124: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg330_1, view_466, permute_245);  arg330_1 = view_466 = permute_245 = None
    view_467: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_124, [1, 128, 2560]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_468: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_467, [1, -1, 32, 80]);  view_467 = None
    permute_246: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    clone_173: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_472: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_173, [32, -1, 80]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_49: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_24, view_472);  div_24 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_475: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_49, [1, 32, 128, 80]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_249: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_176: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_476: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_176, [1, 128, 2560]);  clone_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_477: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_476, [128, 2560]);  view_476 = None
    permute_250: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
    addmm_125: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg332_1, view_477, permute_250);  arg332_1 = view_477 = permute_250 = None
    view_478: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_125, [1, 128, 2560]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_143: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_139, view_478);  add_139 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    sub_64: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_143, getitem_79);  getitem_79 = None
    add_144: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    mul_144: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_39);  sub_64 = rsqrt_39 = None
    mul_145: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_144, arg333_1);  mul_144 = arg333_1 = None
    add_145: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_145, arg334_1);  mul_145 = arg334_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_479: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_145, [128, 2560]);  add_145 = None
    permute_251: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
    addmm_126: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg336_1, view_479, permute_251);  arg336_1 = view_479 = permute_251 = None
    view_480: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_126, [1, 128, 2560]);  addmm_126 = None
    mul_146: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_480, 0.11180339887498948);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_487: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_146, [1, 128, 32, 80]);  mul_146 = None
    permute_256: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None
    clone_180: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_488: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_180, [32, -1, 80]);  clone_180 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_36: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_488, 0);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_481: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_252: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
    addmm_127: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg338_1, view_481, permute_252);  arg338_1 = view_481 = permute_252 = None
    view_482: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_127, [1, 128, 2560]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_483: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_482, [1, -1, 32, 80]);  view_482 = None
    permute_253: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
    clone_178: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_489: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_178, [32, -1, 80]);  clone_178 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_37: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_489, 0);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_484: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_254: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
    addmm_128: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg340_1, view_484, permute_254);  arg340_1 = view_484 = permute_254 = None
    view_485: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_128, [1, 128, 2560]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_486: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_485, [1, -1, 32, 80]);  view_485 = None
    permute_255: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    clone_179: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_490: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_179, [32, -1, 80]);  clone_179 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_38: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_490, 0);  view_490 = None
    _scaled_dot_product_flash_attention_default_12 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_36, unsqueeze_default_37, unsqueeze_default_38, scale = 1.0);  unsqueeze_default_36 = unsqueeze_default_37 = unsqueeze_default_38 = None
    getitem_168: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_12[0];  _scaled_dot_product_flash_attention_default_12 = None
    squeeze_dim_12: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_168, 0);  getitem_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_491: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_12, [1, 32, 128, 80]);  squeeze_dim_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_258: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_182: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_492: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_182, [1, 128, 2560]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_493: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_492, [128, 2560]);  view_492 = None
    permute_259: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
    addmm_129: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg342_1, view_493, permute_259);  arg342_1 = view_493 = permute_259 = None
    view_494: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_129, [1, 128, 2560]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_146: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_143, view_494);  add_143 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    sub_66: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_146, getitem_81);  getitem_81 = None
    add_147: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    mul_147: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_40);  sub_66 = rsqrt_40 = None
    mul_148: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_147, arg343_1);  mul_147 = arg343_1 = None
    add_148: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_148, arg344_1);  mul_148 = arg344_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_495: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_148, [128, 2560]);  add_148 = None
    permute_260: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
    addmm_130: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg346_1, view_495, permute_260);  arg346_1 = view_495 = permute_260 = None
    view_496: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_130, [1, 128, 10240]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_149: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_496, 0.5)
    mul_150: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_496, 0.7071067811865476);  view_496 = None
    erf_13: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_150);  mul_150 = None
    add_149: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_151: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_149, add_149);  mul_149 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_497: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_151, [128, 10240]);  mul_151 = None
    permute_261: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
    addmm_131: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg348_1, view_497, permute_261);  arg348_1 = view_497 = permute_261 = None
    view_498: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_131, [1, 128, 2560]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_150: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_146, view_498);  add_146 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    sub_67: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_150, getitem_83);  getitem_83 = None
    add_151: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    mul_152: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_41);  sub_67 = rsqrt_41 = None
    mul_153: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_152, arg349_1);  mul_152 = arg349_1 = None
    add_152: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_153, arg350_1);  mul_153 = arg350_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_499: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_152, [128, 2560])
    permute_262: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
    addmm_132: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg352_1, view_499, permute_262);  arg352_1 = view_499 = permute_262 = None
    view_500: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_132, [1, 128, 2560]);  addmm_132 = None
    mul_154: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_500, 0.11180339887498948);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_507: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_154, [1, 128, 32, 80]);  mul_154 = None
    permute_267: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
    clone_188: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_508: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_188, [32, -1, 80]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_501: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_152, [128, 2560])
    permute_263: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
    addmm_133: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg354_1, view_501, permute_263);  arg354_1 = view_501 = permute_263 = None
    view_502: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_133, [1, 128, 2560]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_503: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_502, [1, -1, 32, 80]);  view_502 = None
    permute_264: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_503, [0, 2, 1, 3]);  view_503 = None
    clone_186: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_509: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_186, [32, -1, 80]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_268: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_509, [0, 2, 1]);  view_509 = None
    bmm_52: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_508, permute_268);  view_508 = permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_511: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_52, [1, 32, 128, 128]);  bmm_52 = None
    add_153: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_511, expand_1);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_512: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_153, [32, 128, 128]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_26: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_512, [-1], True)
    sub_68: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_512, amax_26);  view_512 = amax_26 = None
    exp_26: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_27: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_26: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_504: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_152, [128, 2560]);  add_152 = None
    permute_265: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
    addmm_134: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg356_1, view_504, permute_265);  arg356_1 = view_504 = permute_265 = None
    view_505: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_134, [1, 128, 2560]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_506: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_505, [1, -1, 32, 80]);  view_505 = None
    permute_266: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None
    clone_187: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_510: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_187, [32, -1, 80]);  clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_53: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_26, view_510);  div_26 = view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_513: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_53, [1, 32, 128, 80]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_269: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_190: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
    view_514: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_190, [1, 128, 2560]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_515: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_514, [128, 2560]);  view_514 = None
    permute_270: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
    addmm_135: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg358_1, view_515, permute_270);  arg358_1 = view_515 = permute_270 = None
    view_516: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_135, [1, 128, 2560]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_154: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_150, view_516);  add_150 = view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    sub_69: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_154, getitem_85);  getitem_85 = None
    add_155: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    mul_155: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_42);  sub_69 = rsqrt_42 = None
    mul_156: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_155, arg359_1);  mul_155 = arg359_1 = None
    add_156: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_156, arg360_1);  mul_156 = arg360_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_517: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_156, [128, 2560]);  add_156 = None
    permute_271: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
    addmm_136: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg362_1, view_517, permute_271);  arg362_1 = view_517 = permute_271 = None
    view_518: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_136, [1, 128, 2560]);  addmm_136 = None
    mul_157: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_518, 0.11180339887498948);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_525: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_157, [1, 128, 32, 80]);  mul_157 = None
    permute_276: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None
    clone_194: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_526: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_194, [32, -1, 80]);  clone_194 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_33: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_526, 0);  view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_519: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_272: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
    addmm_137: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg364_1, view_519, permute_272);  arg364_1 = view_519 = permute_272 = None
    view_520: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_137, [1, 128, 2560]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_521: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_520, [1, -1, 32, 80]);  view_520 = None
    permute_273: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_521, [0, 2, 1, 3]);  view_521 = None
    clone_192: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_527: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_192, [32, -1, 80]);  clone_192 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_34: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_527, 0);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_522: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_274: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg365_1, [1, 0]);  arg365_1 = None
    addmm_138: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg366_1, view_522, permute_274);  arg366_1 = view_522 = permute_274 = None
    view_523: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_138, [1, 128, 2560]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_524: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_523, [1, -1, 32, 80]);  view_523 = None
    permute_275: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
    clone_193: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_528: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_193, [32, -1, 80]);  clone_193 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_35: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_528, 0);  view_528 = None
    _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
    getitem_167: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_11[0];  _scaled_dot_product_flash_attention_default_11 = None
    squeeze_dim_11: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_167, 0);  getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_529: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_11, [1, 32, 128, 80]);  squeeze_dim_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_278: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_196: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
    view_530: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_196, [1, 128, 2560]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_531: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_530, [128, 2560]);  view_530 = None
    permute_279: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
    addmm_139: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg368_1, view_531, permute_279);  arg368_1 = view_531 = permute_279 = None
    view_532: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_139, [1, 128, 2560]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_157: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_154, view_532);  add_154 = view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    sub_71: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_157, getitem_87);  getitem_87 = None
    add_158: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    mul_158: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_43);  sub_71 = rsqrt_43 = None
    mul_159: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_158, arg369_1);  mul_158 = arg369_1 = None
    add_159: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_159, arg370_1);  mul_159 = arg370_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_533: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_159, [128, 2560]);  add_159 = None
    permute_280: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
    addmm_140: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg372_1, view_533, permute_280);  arg372_1 = view_533 = permute_280 = None
    view_534: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_140, [1, 128, 10240]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_160: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_534, 0.5)
    mul_161: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_534, 0.7071067811865476);  view_534 = None
    erf_14: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
    add_160: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_162: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_160, add_160);  mul_160 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_535: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_162, [128, 10240]);  mul_162 = None
    permute_281: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
    addmm_141: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg374_1, view_535, permute_281);  arg374_1 = view_535 = permute_281 = None
    view_536: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_141, [1, 128, 2560]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_161: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_157, view_536);  add_157 = view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    sub_72: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_161, getitem_89);  getitem_89 = None
    add_162: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    mul_163: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_44);  sub_72 = rsqrt_44 = None
    mul_164: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_163, arg375_1);  mul_163 = arg375_1 = None
    add_163: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_164, arg376_1);  mul_164 = arg376_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_537: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_163, [128, 2560])
    permute_282: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
    addmm_142: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg378_1, view_537, permute_282);  arg378_1 = view_537 = permute_282 = None
    view_538: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_142, [1, 128, 2560]);  addmm_142 = None
    mul_165: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_538, 0.11180339887498948);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_545: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_165, [1, 128, 32, 80]);  mul_165 = None
    permute_287: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
    clone_202: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_546: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_202, [32, -1, 80]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_539: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_163, [128, 2560])
    permute_283: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
    addmm_143: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg380_1, view_539, permute_283);  arg380_1 = view_539 = permute_283 = None
    view_540: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_143, [1, 128, 2560]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_541: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_540, [1, -1, 32, 80]);  view_540 = None
    permute_284: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_541, [0, 2, 1, 3]);  view_541 = None
    clone_200: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_547: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_200, [32, -1, 80]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_288: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_547, [0, 2, 1]);  view_547 = None
    bmm_56: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_546, permute_288);  view_546 = permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_549: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_56, [1, 32, 128, 128]);  bmm_56 = None
    add_164: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_549, expand_1);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_550: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_164, [32, 128, 128]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_28: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_550, [-1], True)
    sub_73: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_550, amax_28);  view_550 = amax_28 = None
    exp_28: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_29: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
    div_28: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_542: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_163, [128, 2560]);  add_163 = None
    permute_285: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
    addmm_144: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg382_1, view_542, permute_285);  arg382_1 = view_542 = permute_285 = None
    view_543: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_144, [1, 128, 2560]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_544: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_543, [1, -1, 32, 80]);  view_543 = None
    permute_286: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
    clone_201: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_548: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_201, [32, -1, 80]);  clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_57: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_28, view_548);  div_28 = view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_551: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_57, [1, 32, 128, 80]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_289: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_204: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_552: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_204, [1, 128, 2560]);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_553: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_552, [128, 2560]);  view_552 = None
    permute_290: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
    addmm_145: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg384_1, view_553, permute_290);  arg384_1 = view_553 = permute_290 = None
    view_554: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_145, [1, 128, 2560]);  addmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_165: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_161, view_554);  add_161 = view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    sub_74: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_165, getitem_91);  getitem_91 = None
    add_166: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    mul_166: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_45);  sub_74 = rsqrt_45 = None
    mul_167: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_166, arg385_1);  mul_166 = arg385_1 = None
    add_167: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_167, arg386_1);  mul_167 = arg386_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_555: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_167, [128, 2560]);  add_167 = None
    permute_291: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
    addmm_146: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg388_1, view_555, permute_291);  arg388_1 = view_555 = permute_291 = None
    view_556: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_146, [1, 128, 2560]);  addmm_146 = None
    mul_168: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_556, 0.11180339887498948);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_563: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_168, [1, 128, 32, 80]);  mul_168 = None
    permute_296: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_563, [0, 2, 1, 3]);  view_563 = None
    clone_208: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_564: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_208, [32, -1, 80]);  clone_208 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_30: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_564, 0);  view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_557: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_292: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
    addmm_147: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg390_1, view_557, permute_292);  arg390_1 = view_557 = permute_292 = None
    view_558: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_147, [1, 128, 2560]);  addmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_559: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_558, [1, -1, 32, 80]);  view_558 = None
    permute_293: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None
    clone_206: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_565: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_206, [32, -1, 80]);  clone_206 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_31: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_565, 0);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_560: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_294: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
    addmm_148: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg392_1, view_560, permute_294);  arg392_1 = view_560 = permute_294 = None
    view_561: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_148, [1, 128, 2560]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_562: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_561, [1, -1, 32, 80]);  view_561 = None
    permute_295: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None
    clone_207: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_566: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_207, [32, -1, 80]);  clone_207 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_32: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_566, 0);  view_566 = None
    _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
    getitem_166: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_10[0];  _scaled_dot_product_flash_attention_default_10 = None
    squeeze_dim_10: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_166, 0);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_567: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_10, [1, 32, 128, 80]);  squeeze_dim_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_298: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_567, [0, 2, 1, 3]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_210: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
    view_568: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_210, [1, 128, 2560]);  clone_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_569: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_568, [128, 2560]);  view_568 = None
    permute_299: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
    addmm_149: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg394_1, view_569, permute_299);  arg394_1 = view_569 = permute_299 = None
    view_570: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_149, [1, 128, 2560]);  addmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_168: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_165, view_570);  add_165 = view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    sub_76: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_168, getitem_93);  getitem_93 = None
    add_169: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    mul_169: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_46);  sub_76 = rsqrt_46 = None
    mul_170: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_169, arg395_1);  mul_169 = arg395_1 = None
    add_170: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_170, arg396_1);  mul_170 = arg396_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_571: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_170, [128, 2560]);  add_170 = None
    permute_300: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
    addmm_150: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg398_1, view_571, permute_300);  arg398_1 = view_571 = permute_300 = None
    view_572: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_150, [1, 128, 10240]);  addmm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_171: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_572, 0.5)
    mul_172: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_572, 0.7071067811865476);  view_572 = None
    erf_15: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_171: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_173: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_171, add_171);  mul_171 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_573: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_173, [128, 10240]);  mul_173 = None
    permute_301: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
    addmm_151: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg400_1, view_573, permute_301);  arg400_1 = view_573 = permute_301 = None
    view_574: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_151, [1, 128, 2560]);  addmm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_172: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_168, view_574);  add_168 = view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_172, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    sub_77: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_172, getitem_95);  getitem_95 = None
    add_173: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    mul_174: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_47);  sub_77 = rsqrt_47 = None
    mul_175: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_174, arg401_1);  mul_174 = arg401_1 = None
    add_174: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_175, arg402_1);  mul_175 = arg402_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_575: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_174, [128, 2560])
    permute_302: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
    addmm_152: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg404_1, view_575, permute_302);  arg404_1 = view_575 = permute_302 = None
    view_576: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_152, [1, 128, 2560]);  addmm_152 = None
    mul_176: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_576, 0.11180339887498948);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_583: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_176, [1, 128, 32, 80]);  mul_176 = None
    permute_307: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_583, [0, 2, 1, 3]);  view_583 = None
    clone_216: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_584: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_216, [32, -1, 80]);  clone_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_577: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_174, [128, 2560])
    permute_303: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
    addmm_153: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg406_1, view_577, permute_303);  arg406_1 = view_577 = permute_303 = None
    view_578: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_153, [1, 128, 2560]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_579: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_578, [1, -1, 32, 80]);  view_578 = None
    permute_304: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None
    clone_214: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_585: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_214, [32, -1, 80]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_308: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_585, [0, 2, 1]);  view_585 = None
    bmm_60: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_584, permute_308);  view_584 = permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_587: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_60, [1, 32, 128, 128]);  bmm_60 = None
    add_175: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_587, expand_1);  view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_588: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_175, [32, 128, 128]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_30: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_588, [-1], True)
    sub_78: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_588, amax_30);  view_588 = amax_30 = None
    exp_30: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_78);  sub_78 = None
    sum_31: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
    div_30: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_580: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_174, [128, 2560]);  add_174 = None
    permute_305: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
    addmm_154: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg408_1, view_580, permute_305);  arg408_1 = view_580 = permute_305 = None
    view_581: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_154, [1, 128, 2560]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_582: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_581, [1, -1, 32, 80]);  view_581 = None
    permute_306: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
    clone_215: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_586: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_215, [32, -1, 80]);  clone_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_61: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_30, view_586);  div_30 = view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_589: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_61, [1, 32, 128, 80]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_309: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_218: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    view_590: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_218, [1, 128, 2560]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_591: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_590, [128, 2560]);  view_590 = None
    permute_310: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
    addmm_155: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg410_1, view_591, permute_310);  arg410_1 = view_591 = permute_310 = None
    view_592: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_155, [1, 128, 2560]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_176: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_172, view_592);  add_172 = view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    sub_79: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_176, getitem_97);  getitem_97 = None
    add_177: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    mul_177: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_48);  sub_79 = rsqrt_48 = None
    mul_178: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_177, arg411_1);  mul_177 = arg411_1 = None
    add_178: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_178, arg412_1);  mul_178 = arg412_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_593: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_178, [128, 2560]);  add_178 = None
    permute_311: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
    addmm_156: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg414_1, view_593, permute_311);  arg414_1 = view_593 = permute_311 = None
    view_594: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_156, [1, 128, 2560]);  addmm_156 = None
    mul_179: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_594, 0.11180339887498948);  view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_601: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_179, [1, 128, 32, 80]);  mul_179 = None
    permute_316: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_601, [0, 2, 1, 3]);  view_601 = None
    clone_222: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_602: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_222, [32, -1, 80]);  clone_222 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_27: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_602, 0);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_595: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_312: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
    addmm_157: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg416_1, view_595, permute_312);  arg416_1 = view_595 = permute_312 = None
    view_596: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_157, [1, 128, 2560]);  addmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_597: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_596, [1, -1, 32, 80]);  view_596 = None
    permute_313: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_220: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_603: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_220, [32, -1, 80]);  clone_220 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_28: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_603, 0);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_598: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_314: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
    addmm_158: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg418_1, view_598, permute_314);  arg418_1 = view_598 = permute_314 = None
    view_599: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_158, [1, 128, 2560]);  addmm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_600: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_599, [1, -1, 32, 80]);  view_599 = None
    permute_315: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
    clone_221: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format);  permute_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_604: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_221, [32, -1, 80]);  clone_221 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_29: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_604, 0);  view_604 = None
    _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
    getitem_165: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_9[0];  _scaled_dot_product_flash_attention_default_9 = None
    squeeze_dim_9: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_165, 0);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_605: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_9, [1, 32, 128, 80]);  squeeze_dim_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_318: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_224: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
    view_606: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_224, [1, 128, 2560]);  clone_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_607: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_606, [128, 2560]);  view_606 = None
    permute_319: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
    addmm_159: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg420_1, view_607, permute_319);  arg420_1 = view_607 = permute_319 = None
    view_608: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_159, [1, 128, 2560]);  addmm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_179: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_176, view_608);  add_176 = view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 128, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 128, 1]" = var_mean_49[1];  var_mean_49 = None
    sub_81: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_179, getitem_99);  getitem_99 = None
    add_180: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    mul_180: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_49);  sub_81 = rsqrt_49 = None
    mul_181: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_180, arg421_1);  mul_180 = arg421_1 = None
    add_181: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_181, arg422_1);  mul_181 = arg422_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_609: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_181, [128, 2560]);  add_181 = None
    permute_320: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
    addmm_160: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg424_1, view_609, permute_320);  arg424_1 = view_609 = permute_320 = None
    view_610: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_160, [1, 128, 10240]);  addmm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_182: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_610, 0.5)
    mul_183: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_610, 0.7071067811865476);  view_610 = None
    erf_16: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
    add_182: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_184: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_182, add_182);  mul_182 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_611: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_184, [128, 10240]);  mul_184 = None
    permute_321: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
    addmm_161: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg426_1, view_611, permute_321);  arg426_1 = view_611 = permute_321 = None
    view_612: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_161, [1, 128, 2560]);  addmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_183: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_179, view_612);  add_179 = view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_50 = torch.ops.aten.var_mean.correction(add_183, [2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 128, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 128, 1]" = var_mean_50[1];  var_mean_50 = None
    sub_82: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_183, getitem_101);  getitem_101 = None
    add_184: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    mul_185: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_50);  sub_82 = rsqrt_50 = None
    mul_186: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_185, arg427_1);  mul_185 = arg427_1 = None
    add_185: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_186, arg428_1);  mul_186 = arg428_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_613: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_185, [128, 2560])
    permute_322: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
    addmm_162: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg430_1, view_613, permute_322);  arg430_1 = view_613 = permute_322 = None
    view_614: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_162, [1, 128, 2560]);  addmm_162 = None
    mul_187: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_614, 0.11180339887498948);  view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_621: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_187, [1, 128, 32, 80]);  mul_187 = None
    permute_327: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None
    clone_230: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_622: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_230, [32, -1, 80]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_615: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_185, [128, 2560])
    permute_323: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
    addmm_163: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg432_1, view_615, permute_323);  arg432_1 = view_615 = permute_323 = None
    view_616: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_163, [1, 128, 2560]);  addmm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_617: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_616, [1, -1, 32, 80]);  view_616 = None
    permute_324: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_617, [0, 2, 1, 3]);  view_617 = None
    clone_228: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_324, memory_format = torch.contiguous_format);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_623: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_228, [32, -1, 80]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_328: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_623, [0, 2, 1]);  view_623 = None
    bmm_64: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_622, permute_328);  view_622 = permute_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_625: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_64, [1, 32, 128, 128]);  bmm_64 = None
    add_186: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_625, expand_1);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_626: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_186, [32, 128, 128]);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_32: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_626, [-1], True)
    sub_83: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_626, amax_32);  view_626 = amax_32 = None
    exp_32: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
    sum_33: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
    div_32: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_618: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_185, [128, 2560]);  add_185 = None
    permute_325: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
    addmm_164: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg434_1, view_618, permute_325);  arg434_1 = view_618 = permute_325 = None
    view_619: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_164, [1, 128, 2560]);  addmm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_620: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_619, [1, -1, 32, 80]);  view_619 = None
    permute_326: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None
    clone_229: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_624: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_229, [32, -1, 80]);  clone_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_65: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_32, view_624);  div_32 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_627: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_65, [1, 32, 128, 80]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_329: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_232: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
    view_628: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_232, [1, 128, 2560]);  clone_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_629: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_628, [128, 2560]);  view_628 = None
    permute_330: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
    addmm_165: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg436_1, view_629, permute_330);  arg436_1 = view_629 = permute_330 = None
    view_630: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_165, [1, 128, 2560]);  addmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_187: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_183, view_630);  add_183 = view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_51 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 128, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 128, 1]" = var_mean_51[1];  var_mean_51 = None
    sub_84: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_187, getitem_103);  getitem_103 = None
    add_188: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    mul_188: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_51);  sub_84 = rsqrt_51 = None
    mul_189: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_188, arg437_1);  mul_188 = arg437_1 = None
    add_189: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_189, arg438_1);  mul_189 = arg438_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_631: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_189, [128, 2560]);  add_189 = None
    permute_331: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
    addmm_166: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg440_1, view_631, permute_331);  arg440_1 = view_631 = permute_331 = None
    view_632: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_166, [1, 128, 2560]);  addmm_166 = None
    mul_190: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_632, 0.11180339887498948);  view_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_639: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_190, [1, 128, 32, 80]);  mul_190 = None
    permute_336: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_639, [0, 2, 1, 3]);  view_639 = None
    clone_236: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_640: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_236, [32, -1, 80]);  clone_236 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_24: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_640, 0);  view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_633: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_332: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
    addmm_167: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg442_1, view_633, permute_332);  arg442_1 = view_633 = permute_332 = None
    view_634: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_167, [1, 128, 2560]);  addmm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_635: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_634, [1, -1, 32, 80]);  view_634 = None
    permute_333: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
    clone_234: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_641: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_234, [32, -1, 80]);  clone_234 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_25: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_641, 0);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_636: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_334: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
    addmm_168: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg444_1, view_636, permute_334);  arg444_1 = view_636 = permute_334 = None
    view_637: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_168, [1, 128, 2560]);  addmm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_638: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_637, [1, -1, 32, 80]);  view_637 = None
    permute_335: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
    clone_235: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_642: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_235, [32, -1, 80]);  clone_235 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_26: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_642, 0);  view_642 = None
    _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
    getitem_164: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_8[0];  _scaled_dot_product_flash_attention_default_8 = None
    squeeze_dim_8: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_164, 0);  getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_643: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_8, [1, 32, 128, 80]);  squeeze_dim_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_338: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_643, [0, 2, 1, 3]);  view_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_238: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_338, memory_format = torch.contiguous_format);  permute_338 = None
    view_644: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_238, [1, 128, 2560]);  clone_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_645: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_644, [128, 2560]);  view_644 = None
    permute_339: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
    addmm_169: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg446_1, view_645, permute_339);  arg446_1 = view_645 = permute_339 = None
    view_646: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_169, [1, 128, 2560]);  addmm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_190: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_187, view_646);  add_187 = view_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_52 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 128, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 128, 1]" = var_mean_52[1];  var_mean_52 = None
    sub_86: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_190, getitem_105);  getitem_105 = None
    add_191: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    mul_191: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_52);  sub_86 = rsqrt_52 = None
    mul_192: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_191, arg447_1);  mul_191 = arg447_1 = None
    add_192: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_192, arg448_1);  mul_192 = arg448_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_647: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_192, [128, 2560]);  add_192 = None
    permute_340: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
    addmm_170: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg450_1, view_647, permute_340);  arg450_1 = view_647 = permute_340 = None
    view_648: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_170, [1, 128, 10240]);  addmm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_193: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_648, 0.5)
    mul_194: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_648, 0.7071067811865476);  view_648 = None
    erf_17: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_194);  mul_194 = None
    add_193: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_195: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_193, add_193);  mul_193 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_649: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_195, [128, 10240]);  mul_195 = None
    permute_341: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
    addmm_171: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg452_1, view_649, permute_341);  arg452_1 = view_649 = permute_341 = None
    view_650: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_171, [1, 128, 2560]);  addmm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_194: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_190, view_650);  add_190 = view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_53 = torch.ops.aten.var_mean.correction(add_194, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 128, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 128, 1]" = var_mean_53[1];  var_mean_53 = None
    sub_87: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_194, getitem_107);  getitem_107 = None
    add_195: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    mul_196: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_53);  sub_87 = rsqrt_53 = None
    mul_197: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_196, arg453_1);  mul_196 = arg453_1 = None
    add_196: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_197, arg454_1);  mul_197 = arg454_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_651: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_196, [128, 2560])
    permute_342: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
    addmm_172: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg456_1, view_651, permute_342);  arg456_1 = view_651 = permute_342 = None
    view_652: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_172, [1, 128, 2560]);  addmm_172 = None
    mul_198: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_652, 0.11180339887498948);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_659: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_198, [1, 128, 32, 80]);  mul_198 = None
    permute_347: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_659, [0, 2, 1, 3]);  view_659 = None
    clone_244: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_660: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_244, [32, -1, 80]);  clone_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_653: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_196, [128, 2560])
    permute_343: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
    addmm_173: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg458_1, view_653, permute_343);  arg458_1 = view_653 = permute_343 = None
    view_654: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_173, [1, 128, 2560]);  addmm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_655: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_654, [1, -1, 32, 80]);  view_654 = None
    permute_344: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_655, [0, 2, 1, 3]);  view_655 = None
    clone_242: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_661: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_242, [32, -1, 80]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_348: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_661, [0, 2, 1]);  view_661 = None
    bmm_68: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_660, permute_348);  view_660 = permute_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_663: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_68, [1, 32, 128, 128]);  bmm_68 = None
    add_197: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_663, expand_1);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_664: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_197, [32, 128, 128]);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_34: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_664, [-1], True)
    sub_88: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_664, amax_34);  view_664 = amax_34 = None
    exp_34: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_88);  sub_88 = None
    sum_35: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
    div_34: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_656: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_196, [128, 2560]);  add_196 = None
    permute_345: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
    addmm_174: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg460_1, view_656, permute_345);  arg460_1 = view_656 = permute_345 = None
    view_657: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_174, [1, 128, 2560]);  addmm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_658: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_657, [1, -1, 32, 80]);  view_657 = None
    permute_346: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
    clone_243: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_662: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_243, [32, -1, 80]);  clone_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_69: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_34, view_662);  div_34 = view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_665: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_69, [1, 32, 128, 80]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_349: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_665, [0, 2, 1, 3]);  view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_246: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_349, memory_format = torch.contiguous_format);  permute_349 = None
    view_666: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_246, [1, 128, 2560]);  clone_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_667: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_666, [128, 2560]);  view_666 = None
    permute_350: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
    addmm_175: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg462_1, view_667, permute_350);  arg462_1 = view_667 = permute_350 = None
    view_668: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_175, [1, 128, 2560]);  addmm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_198: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_194, view_668);  add_194 = view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_54 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 128, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 128, 1]" = var_mean_54[1];  var_mean_54 = None
    sub_89: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_198, getitem_109);  getitem_109 = None
    add_199: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    mul_199: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_54);  sub_89 = rsqrt_54 = None
    mul_200: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_199, arg463_1);  mul_199 = arg463_1 = None
    add_200: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_200, arg464_1);  mul_200 = arg464_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_669: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_200, [128, 2560]);  add_200 = None
    permute_351: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
    addmm_176: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg466_1, view_669, permute_351);  arg466_1 = view_669 = permute_351 = None
    view_670: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_176, [1, 128, 2560]);  addmm_176 = None
    mul_201: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_670, 0.11180339887498948);  view_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_677: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_201, [1, 128, 32, 80]);  mul_201 = None
    permute_356: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_677, [0, 2, 1, 3]);  view_677 = None
    clone_250: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_678: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_250, [32, -1, 80]);  clone_250 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_21: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_678, 0);  view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_671: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_352: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
    addmm_177: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg468_1, view_671, permute_352);  arg468_1 = view_671 = permute_352 = None
    view_672: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_177, [1, 128, 2560]);  addmm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_673: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_672, [1, -1, 32, 80]);  view_672 = None
    permute_353: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
    clone_248: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_679: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_248, [32, -1, 80]);  clone_248 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_22: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_679, 0);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_674: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_354: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
    addmm_178: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg470_1, view_674, permute_354);  arg470_1 = view_674 = permute_354 = None
    view_675: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_178, [1, 128, 2560]);  addmm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_676: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_675, [1, -1, 32, 80]);  view_675 = None
    permute_355: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None
    clone_249: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_680: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_249, [32, -1, 80]);  clone_249 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_23: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_680, 0);  view_680 = None
    _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
    getitem_163: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_7[0];  _scaled_dot_product_flash_attention_default_7 = None
    squeeze_dim_7: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_163, 0);  getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_681: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_7, [1, 32, 128, 80]);  squeeze_dim_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_358: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_252: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_682: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_252, [1, 128, 2560]);  clone_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_683: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_682, [128, 2560]);  view_682 = None
    permute_359: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
    addmm_179: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg472_1, view_683, permute_359);  arg472_1 = view_683 = permute_359 = None
    view_684: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_179, [1, 128, 2560]);  addmm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_201: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_198, view_684);  add_198 = view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_55 = torch.ops.aten.var_mean.correction(add_201, [2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 128, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 128, 1]" = var_mean_55[1];  var_mean_55 = None
    sub_91: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_201, getitem_111);  getitem_111 = None
    add_202: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    mul_202: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_55);  sub_91 = rsqrt_55 = None
    mul_203: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_202, arg473_1);  mul_202 = arg473_1 = None
    add_203: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_203, arg474_1);  mul_203 = arg474_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_685: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_203, [128, 2560]);  add_203 = None
    permute_360: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
    addmm_180: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg476_1, view_685, permute_360);  arg476_1 = view_685 = permute_360 = None
    view_686: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_180, [1, 128, 10240]);  addmm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_204: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_686, 0.5)
    mul_205: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_686, 0.7071067811865476);  view_686 = None
    erf_18: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_205);  mul_205 = None
    add_204: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_206: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_204, add_204);  mul_204 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_687: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_206, [128, 10240]);  mul_206 = None
    permute_361: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
    addmm_181: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg478_1, view_687, permute_361);  arg478_1 = view_687 = permute_361 = None
    view_688: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_181, [1, 128, 2560]);  addmm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_205: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_201, view_688);  add_201 = view_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_56 = torch.ops.aten.var_mean.correction(add_205, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 128, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 128, 1]" = var_mean_56[1];  var_mean_56 = None
    sub_92: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_205, getitem_113);  getitem_113 = None
    add_206: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    mul_207: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_56);  sub_92 = rsqrt_56 = None
    mul_208: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_207, arg479_1);  mul_207 = arg479_1 = None
    add_207: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_208, arg480_1);  mul_208 = arg480_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_689: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_207, [128, 2560])
    permute_362: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
    addmm_182: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg482_1, view_689, permute_362);  arg482_1 = view_689 = permute_362 = None
    view_690: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_182, [1, 128, 2560]);  addmm_182 = None
    mul_209: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_690, 0.11180339887498948);  view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_697: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_209, [1, 128, 32, 80]);  mul_209 = None
    permute_367: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None
    clone_258: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_698: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_258, [32, -1, 80]);  clone_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_691: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_207, [128, 2560])
    permute_363: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
    addmm_183: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg484_1, view_691, permute_363);  arg484_1 = view_691 = permute_363 = None
    view_692: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_183, [1, 128, 2560]);  addmm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_693: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_692, [1, -1, 32, 80]);  view_692 = None
    permute_364: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
    clone_256: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_699: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_256, [32, -1, 80]);  clone_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_368: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_699, [0, 2, 1]);  view_699 = None
    bmm_72: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_698, permute_368);  view_698 = permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_701: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_72, [1, 32, 128, 128]);  bmm_72 = None
    add_208: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_701, expand_1);  view_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_702: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_208, [32, 128, 128]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_36: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_702, [-1], True)
    sub_93: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_702, amax_36);  view_702 = amax_36 = None
    exp_36: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
    sum_37: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
    div_36: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_694: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_207, [128, 2560]);  add_207 = None
    permute_365: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg485_1, [1, 0]);  arg485_1 = None
    addmm_184: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg486_1, view_694, permute_365);  arg486_1 = view_694 = permute_365 = None
    view_695: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_184, [1, 128, 2560]);  addmm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_696: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_695, [1, -1, 32, 80]);  view_695 = None
    permute_366: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None
    clone_257: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_700: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_257, [32, -1, 80]);  clone_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_73: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_36, view_700);  div_36 = view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_703: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_73, [1, 32, 128, 80]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_369: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_703, [0, 2, 1, 3]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_260: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
    view_704: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_260, [1, 128, 2560]);  clone_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_705: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_704, [128, 2560]);  view_704 = None
    permute_370: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
    addmm_185: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg488_1, view_705, permute_370);  arg488_1 = view_705 = permute_370 = None
    view_706: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_185, [1, 128, 2560]);  addmm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_209: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_205, view_706);  add_205 = view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_57 = torch.ops.aten.var_mean.correction(add_209, [2], correction = 0, keepdim = True)
    getitem_114: "f32[1, 128, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 128, 1]" = var_mean_57[1];  var_mean_57 = None
    sub_94: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_209, getitem_115);  getitem_115 = None
    add_210: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_57: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    mul_210: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_57);  sub_94 = rsqrt_57 = None
    mul_211: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_210, arg489_1);  mul_210 = arg489_1 = None
    add_211: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_211, arg490_1);  mul_211 = arg490_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_707: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_211, [128, 2560]);  add_211 = None
    permute_371: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
    addmm_186: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg492_1, view_707, permute_371);  arg492_1 = view_707 = permute_371 = None
    view_708: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_186, [1, 128, 2560]);  addmm_186 = None
    mul_212: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_708, 0.11180339887498948);  view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_715: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_212, [1, 128, 32, 80]);  mul_212 = None
    permute_376: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_715, [0, 2, 1, 3]);  view_715 = None
    clone_264: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_716: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_264, [32, -1, 80]);  clone_264 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_18: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_716, 0);  view_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_709: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_372: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
    addmm_187: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg494_1, view_709, permute_372);  arg494_1 = view_709 = permute_372 = None
    view_710: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_187, [1, 128, 2560]);  addmm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_711: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_710, [1, -1, 32, 80]);  view_710 = None
    permute_373: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_711, [0, 2, 1, 3]);  view_711 = None
    clone_262: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_717: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_262, [32, -1, 80]);  clone_262 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_19: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_717, 0);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_712: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_374: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
    addmm_188: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg496_1, view_712, permute_374);  arg496_1 = view_712 = permute_374 = None
    view_713: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_188, [1, 128, 2560]);  addmm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_714: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_713, [1, -1, 32, 80]);  view_713 = None
    permute_375: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None
    clone_263: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_718: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_263, [32, -1, 80]);  clone_263 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_20: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_718, 0);  view_718 = None
    _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
    getitem_162: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_6[0];  _scaled_dot_product_flash_attention_default_6 = None
    squeeze_dim_6: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_162, 0);  getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_719: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_6, [1, 32, 128, 80]);  squeeze_dim_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_378: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_719, [0, 2, 1, 3]);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_266: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format);  permute_378 = None
    view_720: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_266, [1, 128, 2560]);  clone_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_721: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_720, [128, 2560]);  view_720 = None
    permute_379: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
    addmm_189: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg498_1, view_721, permute_379);  arg498_1 = view_721 = permute_379 = None
    view_722: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_189, [1, 128, 2560]);  addmm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_212: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_209, view_722);  add_209 = view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_58 = torch.ops.aten.var_mean.correction(add_212, [2], correction = 0, keepdim = True)
    getitem_116: "f32[1, 128, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 128, 1]" = var_mean_58[1];  var_mean_58 = None
    sub_96: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_212, getitem_117);  getitem_117 = None
    add_213: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_58: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    mul_213: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_58);  sub_96 = rsqrt_58 = None
    mul_214: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_213, arg499_1);  mul_213 = arg499_1 = None
    add_214: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_214, arg500_1);  mul_214 = arg500_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_723: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_214, [128, 2560]);  add_214 = None
    permute_380: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
    addmm_190: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg502_1, view_723, permute_380);  arg502_1 = view_723 = permute_380 = None
    view_724: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_190, [1, 128, 10240]);  addmm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_215: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_724, 0.5)
    mul_216: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_724, 0.7071067811865476);  view_724 = None
    erf_19: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
    add_215: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_217: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_215, add_215);  mul_215 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_725: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_217, [128, 10240]);  mul_217 = None
    permute_381: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg503_1, [1, 0]);  arg503_1 = None
    addmm_191: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg504_1, view_725, permute_381);  arg504_1 = view_725 = permute_381 = None
    view_726: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_191, [1, 128, 2560]);  addmm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_216: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_212, view_726);  add_212 = view_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_59 = torch.ops.aten.var_mean.correction(add_216, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 128, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 128, 1]" = var_mean_59[1];  var_mean_59 = None
    sub_97: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_216, getitem_119);  getitem_119 = None
    add_217: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    mul_218: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_59);  sub_97 = rsqrt_59 = None
    mul_219: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_218, arg505_1);  mul_218 = arg505_1 = None
    add_218: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_219, arg506_1);  mul_219 = arg506_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_727: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_218, [128, 2560])
    permute_382: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
    addmm_192: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg508_1, view_727, permute_382);  arg508_1 = view_727 = permute_382 = None
    view_728: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_192, [1, 128, 2560]);  addmm_192 = None
    mul_220: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_728, 0.11180339887498948);  view_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_735: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_220, [1, 128, 32, 80]);  mul_220 = None
    permute_387: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None
    clone_272: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_736: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_272, [32, -1, 80]);  clone_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_729: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_218, [128, 2560])
    permute_383: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
    addmm_193: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg510_1, view_729, permute_383);  arg510_1 = view_729 = permute_383 = None
    view_730: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_193, [1, 128, 2560]);  addmm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_731: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_730, [1, -1, 32, 80]);  view_730 = None
    permute_384: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None
    clone_270: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_737: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_270, [32, -1, 80]);  clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_388: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_737, [0, 2, 1]);  view_737 = None
    bmm_76: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_736, permute_388);  view_736 = permute_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_739: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_76, [1, 32, 128, 128]);  bmm_76 = None
    add_219: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_739, expand_1);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_740: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_219, [32, 128, 128]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_38: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_740, [-1], True)
    sub_98: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_740, amax_38);  view_740 = amax_38 = None
    exp_38: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_98);  sub_98 = None
    sum_39: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
    div_38: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_732: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_218, [128, 2560]);  add_218 = None
    permute_385: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
    addmm_194: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg512_1, view_732, permute_385);  arg512_1 = view_732 = permute_385 = None
    view_733: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_194, [1, 128, 2560]);  addmm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_734: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_733, [1, -1, 32, 80]);  view_733 = None
    permute_386: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
    clone_271: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_738: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_271, [32, -1, 80]);  clone_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_77: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_38, view_738);  div_38 = view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_741: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_77, [1, 32, 128, 80]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_389: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_741, [0, 2, 1, 3]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_274: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    view_742: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_274, [1, 128, 2560]);  clone_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_743: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_742, [128, 2560]);  view_742 = None
    permute_390: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
    addmm_195: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg514_1, view_743, permute_390);  arg514_1 = view_743 = permute_390 = None
    view_744: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_195, [1, 128, 2560]);  addmm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_220: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_216, view_744);  add_216 = view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_60 = torch.ops.aten.var_mean.correction(add_220, [2], correction = 0, keepdim = True)
    getitem_120: "f32[1, 128, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 128, 1]" = var_mean_60[1];  var_mean_60 = None
    sub_99: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_220, getitem_121);  getitem_121 = None
    add_221: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_60: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    mul_221: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_60);  sub_99 = rsqrt_60 = None
    mul_222: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_221, arg515_1);  mul_221 = arg515_1 = None
    add_222: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_222, arg516_1);  mul_222 = arg516_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_745: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_222, [128, 2560]);  add_222 = None
    permute_391: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg517_1, [1, 0]);  arg517_1 = None
    addmm_196: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg518_1, view_745, permute_391);  arg518_1 = view_745 = permute_391 = None
    view_746: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_196, [1, 128, 2560]);  addmm_196 = None
    mul_223: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_746, 0.11180339887498948);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_753: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_223, [1, 128, 32, 80]);  mul_223 = None
    permute_396: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_753, [0, 2, 1, 3]);  view_753 = None
    clone_278: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_396, memory_format = torch.contiguous_format);  permute_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_754: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_278, [32, -1, 80]);  clone_278 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_15: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_754, 0);  view_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_747: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_392: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
    addmm_197: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg520_1, view_747, permute_392);  arg520_1 = view_747 = permute_392 = None
    view_748: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_197, [1, 128, 2560]);  addmm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_749: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_748, [1, -1, 32, 80]);  view_748 = None
    permute_393: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_749, [0, 2, 1, 3]);  view_749 = None
    clone_276: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_755: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_276, [32, -1, 80]);  clone_276 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_16: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_755, 0);  view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_750: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_394: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg521_1, [1, 0]);  arg521_1 = None
    addmm_198: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg522_1, view_750, permute_394);  arg522_1 = view_750 = permute_394 = None
    view_751: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_198, [1, 128, 2560]);  addmm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_752: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_751, [1, -1, 32, 80]);  view_751 = None
    permute_395: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_752, [0, 2, 1, 3]);  view_752 = None
    clone_277: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_756: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_277, [32, -1, 80]);  clone_277 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_17: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_756, 0);  view_756 = None
    _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
    getitem_161: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_5[0];  _scaled_dot_product_flash_attention_default_5 = None
    squeeze_dim_5: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_161, 0);  getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_757: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_5, [1, 32, 128, 80]);  squeeze_dim_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_398: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_757, [0, 2, 1, 3]);  view_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_280: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_398, memory_format = torch.contiguous_format);  permute_398 = None
    view_758: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_280, [1, 128, 2560]);  clone_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_759: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_758, [128, 2560]);  view_758 = None
    permute_399: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg523_1, [1, 0]);  arg523_1 = None
    addmm_199: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg524_1, view_759, permute_399);  arg524_1 = view_759 = permute_399 = None
    view_760: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_199, [1, 128, 2560]);  addmm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_223: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_220, view_760);  add_220 = view_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_61 = torch.ops.aten.var_mean.correction(add_223, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 128, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 128, 1]" = var_mean_61[1];  var_mean_61 = None
    sub_101: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_223, getitem_123);  getitem_123 = None
    add_224: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_61: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    mul_224: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_61);  sub_101 = rsqrt_61 = None
    mul_225: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_224, arg525_1);  mul_224 = arg525_1 = None
    add_225: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_225, arg526_1);  mul_225 = arg526_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_761: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_225, [128, 2560]);  add_225 = None
    permute_400: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg527_1, [1, 0]);  arg527_1 = None
    addmm_200: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg528_1, view_761, permute_400);  arg528_1 = view_761 = permute_400 = None
    view_762: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_200, [1, 128, 10240]);  addmm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_226: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_762, 0.5)
    mul_227: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_762, 0.7071067811865476);  view_762 = None
    erf_20: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_226: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_228: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_226, add_226);  mul_226 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_763: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_228, [128, 10240]);  mul_228 = None
    permute_401: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg529_1, [1, 0]);  arg529_1 = None
    addmm_201: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg530_1, view_763, permute_401);  arg530_1 = view_763 = permute_401 = None
    view_764: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_201, [1, 128, 2560]);  addmm_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_227: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_223, view_764);  add_223 = view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_62 = torch.ops.aten.var_mean.correction(add_227, [2], correction = 0, keepdim = True)
    getitem_124: "f32[1, 128, 1]" = var_mean_62[0]
    getitem_125: "f32[1, 128, 1]" = var_mean_62[1];  var_mean_62 = None
    sub_102: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_227, getitem_125);  getitem_125 = None
    add_228: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_62: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    mul_229: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_62);  sub_102 = rsqrt_62 = None
    mul_230: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_229, arg531_1);  mul_229 = arg531_1 = None
    add_229: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_230, arg532_1);  mul_230 = arg532_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_765: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_229, [128, 2560])
    permute_402: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg533_1, [1, 0]);  arg533_1 = None
    addmm_202: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg534_1, view_765, permute_402);  arg534_1 = view_765 = permute_402 = None
    view_766: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_202, [1, 128, 2560]);  addmm_202 = None
    mul_231: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_766, 0.11180339887498948);  view_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_773: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_231, [1, 128, 32, 80]);  mul_231 = None
    permute_407: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None
    clone_286: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_774: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_286, [32, -1, 80]);  clone_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_767: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_229, [128, 2560])
    permute_403: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
    addmm_203: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg536_1, view_767, permute_403);  arg536_1 = view_767 = permute_403 = None
    view_768: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_203, [1, 128, 2560]);  addmm_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_769: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_768, [1, -1, 32, 80]);  view_768 = None
    permute_404: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_769, [0, 2, 1, 3]);  view_769 = None
    clone_284: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_775: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_284, [32, -1, 80]);  clone_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_408: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_775, [0, 2, 1]);  view_775 = None
    bmm_80: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_774, permute_408);  view_774 = permute_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_777: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_80, [1, 32, 128, 128]);  bmm_80 = None
    add_230: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_777, expand_1);  view_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_778: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_230, [32, 128, 128]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_40: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_778, [-1], True)
    sub_103: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_778, amax_40);  view_778 = amax_40 = None
    exp_40: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_103);  sub_103 = None
    sum_41: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
    div_40: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_770: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_229, [128, 2560]);  add_229 = None
    permute_405: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg537_1, [1, 0]);  arg537_1 = None
    addmm_204: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg538_1, view_770, permute_405);  arg538_1 = view_770 = permute_405 = None
    view_771: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_204, [1, 128, 2560]);  addmm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_772: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_771, [1, -1, 32, 80]);  view_771 = None
    permute_406: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
    clone_285: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_406, memory_format = torch.contiguous_format);  permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_776: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_285, [32, -1, 80]);  clone_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_81: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_40, view_776);  div_40 = view_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_779: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_81, [1, 32, 128, 80]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_409: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_779, [0, 2, 1, 3]);  view_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_288: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    view_780: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_288, [1, 128, 2560]);  clone_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_781: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_780, [128, 2560]);  view_780 = None
    permute_410: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg539_1, [1, 0]);  arg539_1 = None
    addmm_205: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg540_1, view_781, permute_410);  arg540_1 = view_781 = permute_410 = None
    view_782: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_205, [1, 128, 2560]);  addmm_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_231: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_227, view_782);  add_227 = view_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_63 = torch.ops.aten.var_mean.correction(add_231, [2], correction = 0, keepdim = True)
    getitem_126: "f32[1, 128, 1]" = var_mean_63[0]
    getitem_127: "f32[1, 128, 1]" = var_mean_63[1];  var_mean_63 = None
    sub_104: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_231, getitem_127);  getitem_127 = None
    add_232: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_63: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
    mul_232: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_63);  sub_104 = rsqrt_63 = None
    mul_233: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_232, arg541_1);  mul_232 = arg541_1 = None
    add_233: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_233, arg542_1);  mul_233 = arg542_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_783: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_233, [128, 2560]);  add_233 = None
    permute_411: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg543_1, [1, 0]);  arg543_1 = None
    addmm_206: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg544_1, view_783, permute_411);  arg544_1 = view_783 = permute_411 = None
    view_784: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_206, [1, 128, 2560]);  addmm_206 = None
    mul_234: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_784, 0.11180339887498948);  view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_791: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_234, [1, 128, 32, 80]);  mul_234 = None
    permute_416: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_791, [0, 2, 1, 3]);  view_791 = None
    clone_292: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_416, memory_format = torch.contiguous_format);  permute_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_792: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_292, [32, -1, 80]);  clone_292 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_12: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_792, 0);  view_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_785: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_412: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg545_1, [1, 0]);  arg545_1 = None
    addmm_207: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg546_1, view_785, permute_412);  arg546_1 = view_785 = permute_412 = None
    view_786: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_207, [1, 128, 2560]);  addmm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_787: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_786, [1, -1, 32, 80]);  view_786 = None
    permute_413: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
    clone_290: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_793: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_290, [32, -1, 80]);  clone_290 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_13: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_793, 0);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_788: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_414: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
    addmm_208: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg548_1, view_788, permute_414);  arg548_1 = view_788 = permute_414 = None
    view_789: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_208, [1, 128, 2560]);  addmm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_790: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_789, [1, -1, 32, 80]);  view_789 = None
    permute_415: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_790, [0, 2, 1, 3]);  view_790 = None
    clone_291: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_415, memory_format = torch.contiguous_format);  permute_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_794: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_291, [32, -1, 80]);  clone_291 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_14: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_794, 0);  view_794 = None
    _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
    getitem_160: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_4[0];  _scaled_dot_product_flash_attention_default_4 = None
    squeeze_dim_4: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_160, 0);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_795: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_4, [1, 32, 128, 80]);  squeeze_dim_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_418: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_795, [0, 2, 1, 3]);  view_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_294: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
    view_796: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_294, [1, 128, 2560]);  clone_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_797: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_796, [128, 2560]);  view_796 = None
    permute_419: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg549_1, [1, 0]);  arg549_1 = None
    addmm_209: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg550_1, view_797, permute_419);  arg550_1 = view_797 = permute_419 = None
    view_798: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_209, [1, 128, 2560]);  addmm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_234: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_231, view_798);  add_231 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_64 = torch.ops.aten.var_mean.correction(add_234, [2], correction = 0, keepdim = True)
    getitem_128: "f32[1, 128, 1]" = var_mean_64[0]
    getitem_129: "f32[1, 128, 1]" = var_mean_64[1];  var_mean_64 = None
    sub_106: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_234, getitem_129);  getitem_129 = None
    add_235: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_64: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    mul_235: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_64);  sub_106 = rsqrt_64 = None
    mul_236: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_235, arg551_1);  mul_235 = arg551_1 = None
    add_236: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_236, arg552_1);  mul_236 = arg552_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_799: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_236, [128, 2560]);  add_236 = None
    permute_420: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg553_1, [1, 0]);  arg553_1 = None
    addmm_210: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg554_1, view_799, permute_420);  arg554_1 = view_799 = permute_420 = None
    view_800: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_210, [1, 128, 10240]);  addmm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_237: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_800, 0.5)
    mul_238: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_800, 0.7071067811865476);  view_800 = None
    erf_21: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
    add_237: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_239: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_237, add_237);  mul_237 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_801: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_239, [128, 10240]);  mul_239 = None
    permute_421: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg555_1, [1, 0]);  arg555_1 = None
    addmm_211: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg556_1, view_801, permute_421);  arg556_1 = view_801 = permute_421 = None
    view_802: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_211, [1, 128, 2560]);  addmm_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_238: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_234, view_802);  add_234 = view_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_65 = torch.ops.aten.var_mean.correction(add_238, [2], correction = 0, keepdim = True)
    getitem_130: "f32[1, 128, 1]" = var_mean_65[0]
    getitem_131: "f32[1, 128, 1]" = var_mean_65[1];  var_mean_65 = None
    sub_107: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_238, getitem_131);  getitem_131 = None
    add_239: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_65: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    mul_240: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_65);  sub_107 = rsqrt_65 = None
    mul_241: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_240, arg557_1);  mul_240 = arg557_1 = None
    add_240: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_241, arg558_1);  mul_241 = arg558_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_803: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_240, [128, 2560])
    permute_422: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg559_1, [1, 0]);  arg559_1 = None
    addmm_212: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg560_1, view_803, permute_422);  arg560_1 = view_803 = permute_422 = None
    view_804: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_212, [1, 128, 2560]);  addmm_212 = None
    mul_242: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_804, 0.11180339887498948);  view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_811: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_242, [1, 128, 32, 80]);  mul_242 = None
    permute_427: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None
    clone_300: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_812: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_300, [32, -1, 80]);  clone_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_805: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_240, [128, 2560])
    permute_423: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg561_1, [1, 0]);  arg561_1 = None
    addmm_213: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg562_1, view_805, permute_423);  arg562_1 = view_805 = permute_423 = None
    view_806: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_213, [1, 128, 2560]);  addmm_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_807: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_806, [1, -1, 32, 80]);  view_806 = None
    permute_424: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_807, [0, 2, 1, 3]);  view_807 = None
    clone_298: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_813: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_298, [32, -1, 80]);  clone_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_428: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_813, [0, 2, 1]);  view_813 = None
    bmm_84: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_812, permute_428);  view_812 = permute_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_815: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_84, [1, 32, 128, 128]);  bmm_84 = None
    add_241: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_815, expand_1);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_816: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_241, [32, 128, 128]);  add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_42: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_816, [-1], True)
    sub_108: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_816, amax_42);  view_816 = amax_42 = None
    exp_42: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_108);  sub_108 = None
    sum_43: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
    div_42: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_808: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_240, [128, 2560]);  add_240 = None
    permute_425: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg563_1, [1, 0]);  arg563_1 = None
    addmm_214: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg564_1, view_808, permute_425);  arg564_1 = view_808 = permute_425 = None
    view_809: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_214, [1, 128, 2560]);  addmm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_810: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_809, [1, -1, 32, 80]);  view_809 = None
    permute_426: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_810, [0, 2, 1, 3]);  view_810 = None
    clone_299: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_814: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_299, [32, -1, 80]);  clone_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_85: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_42, view_814);  div_42 = view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_817: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_85, [1, 32, 128, 80]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_429: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_817, [0, 2, 1, 3]);  view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_302: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_429, memory_format = torch.contiguous_format);  permute_429 = None
    view_818: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_302, [1, 128, 2560]);  clone_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_819: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_818, [128, 2560]);  view_818 = None
    permute_430: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
    addmm_215: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg566_1, view_819, permute_430);  arg566_1 = view_819 = permute_430 = None
    view_820: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_215, [1, 128, 2560]);  addmm_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_242: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_238, view_820);  add_238 = view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_66 = torch.ops.aten.var_mean.correction(add_242, [2], correction = 0, keepdim = True)
    getitem_132: "f32[1, 128, 1]" = var_mean_66[0]
    getitem_133: "f32[1, 128, 1]" = var_mean_66[1];  var_mean_66 = None
    sub_109: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_242, getitem_133);  getitem_133 = None
    add_243: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_66: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
    mul_243: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_66);  sub_109 = rsqrt_66 = None
    mul_244: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_243, arg567_1);  mul_243 = arg567_1 = None
    add_244: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_244, arg568_1);  mul_244 = arg568_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_821: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_244, [128, 2560]);  add_244 = None
    permute_431: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg569_1, [1, 0]);  arg569_1 = None
    addmm_216: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg570_1, view_821, permute_431);  arg570_1 = view_821 = permute_431 = None
    view_822: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_216, [1, 128, 2560]);  addmm_216 = None
    mul_245: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_822, 0.11180339887498948);  view_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_829: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_245, [1, 128, 32, 80]);  mul_245 = None
    permute_436: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_829, [0, 2, 1, 3]);  view_829 = None
    clone_306: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_830: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_306, [32, -1, 80]);  clone_306 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_9: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_830, 0);  view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_823: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_432: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
    addmm_217: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg572_1, view_823, permute_432);  arg572_1 = view_823 = permute_432 = None
    view_824: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_217, [1, 128, 2560]);  addmm_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_825: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_824, [1, -1, 32, 80]);  view_824 = None
    permute_433: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_825, [0, 2, 1, 3]);  view_825 = None
    clone_304: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_831: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_304, [32, -1, 80]);  clone_304 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_10: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_831, 0);  view_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_826: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_434: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg573_1, [1, 0]);  arg573_1 = None
    addmm_218: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg574_1, view_826, permute_434);  arg574_1 = view_826 = permute_434 = None
    view_827: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_218, [1, 128, 2560]);  addmm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_828: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_827, [1, -1, 32, 80]);  view_827 = None
    permute_435: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_828, [0, 2, 1, 3]);  view_828 = None
    clone_305: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_832: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_305, [32, -1, 80]);  clone_305 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_11: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_832, 0);  view_832 = None
    _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
    getitem_159: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_3[0];  _scaled_dot_product_flash_attention_default_3 = None
    squeeze_dim_3: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_159, 0);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_833: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_3, [1, 32, 128, 80]);  squeeze_dim_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_438: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_308: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_438, memory_format = torch.contiguous_format);  permute_438 = None
    view_834: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_308, [1, 128, 2560]);  clone_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_835: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_834, [128, 2560]);  view_834 = None
    permute_439: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg575_1, [1, 0]);  arg575_1 = None
    addmm_219: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg576_1, view_835, permute_439);  arg576_1 = view_835 = permute_439 = None
    view_836: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_219, [1, 128, 2560]);  addmm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_245: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_242, view_836);  add_242 = view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_67 = torch.ops.aten.var_mean.correction(add_245, [2], correction = 0, keepdim = True)
    getitem_134: "f32[1, 128, 1]" = var_mean_67[0]
    getitem_135: "f32[1, 128, 1]" = var_mean_67[1];  var_mean_67 = None
    sub_111: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_245, getitem_135);  getitem_135 = None
    add_246: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
    rsqrt_67: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
    mul_246: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_67);  sub_111 = rsqrt_67 = None
    mul_247: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_246, arg577_1);  mul_246 = arg577_1 = None
    add_247: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_247, arg578_1);  mul_247 = arg578_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_837: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_247, [128, 2560]);  add_247 = None
    permute_440: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg579_1, [1, 0]);  arg579_1 = None
    addmm_220: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg580_1, view_837, permute_440);  arg580_1 = view_837 = permute_440 = None
    view_838: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_220, [1, 128, 10240]);  addmm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_248: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_838, 0.5)
    mul_249: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_838, 0.7071067811865476);  view_838 = None
    erf_22: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_249);  mul_249 = None
    add_248: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_250: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_248, add_248);  mul_248 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_839: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_250, [128, 10240]);  mul_250 = None
    permute_441: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg581_1, [1, 0]);  arg581_1 = None
    addmm_221: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg582_1, view_839, permute_441);  arg582_1 = view_839 = permute_441 = None
    view_840: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_221, [1, 128, 2560]);  addmm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_249: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_245, view_840);  add_245 = view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_68 = torch.ops.aten.var_mean.correction(add_249, [2], correction = 0, keepdim = True)
    getitem_136: "f32[1, 128, 1]" = var_mean_68[0]
    getitem_137: "f32[1, 128, 1]" = var_mean_68[1];  var_mean_68 = None
    sub_112: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_249, getitem_137);  getitem_137 = None
    add_250: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_68: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
    mul_251: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_68);  sub_112 = rsqrt_68 = None
    mul_252: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_251, arg583_1);  mul_251 = arg583_1 = None
    add_251: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_252, arg584_1);  mul_252 = arg584_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_841: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_251, [128, 2560])
    permute_442: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg585_1, [1, 0]);  arg585_1 = None
    addmm_222: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg586_1, view_841, permute_442);  arg586_1 = view_841 = permute_442 = None
    view_842: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_222, [1, 128, 2560]);  addmm_222 = None
    mul_253: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_842, 0.11180339887498948);  view_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_849: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_253, [1, 128, 32, 80]);  mul_253 = None
    permute_447: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_849, [0, 2, 1, 3]);  view_849 = None
    clone_314: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_850: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_314, [32, -1, 80]);  clone_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_843: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_251, [128, 2560])
    permute_443: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
    addmm_223: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg588_1, view_843, permute_443);  arg588_1 = view_843 = permute_443 = None
    view_844: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_223, [1, 128, 2560]);  addmm_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_845: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_844, [1, -1, 32, 80]);  view_844 = None
    permute_444: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_845, [0, 2, 1, 3]);  view_845 = None
    clone_312: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_444, memory_format = torch.contiguous_format);  permute_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_851: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_312, [32, -1, 80]);  clone_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_448: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_851, [0, 2, 1]);  view_851 = None
    bmm_88: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_850, permute_448);  view_850 = permute_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_853: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_88, [1, 32, 128, 128]);  bmm_88 = None
    add_252: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_853, expand_1);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_854: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_252, [32, 128, 128]);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_44: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_854, [-1], True)
    sub_113: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_854, amax_44);  view_854 = amax_44 = None
    exp_44: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_113);  sub_113 = None
    sum_45: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
    div_44: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_846: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_251, [128, 2560]);  add_251 = None
    permute_445: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg589_1, [1, 0]);  arg589_1 = None
    addmm_224: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg590_1, view_846, permute_445);  arg590_1 = view_846 = permute_445 = None
    view_847: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_224, [1, 128, 2560]);  addmm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_848: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_847, [1, -1, 32, 80]);  view_847 = None
    permute_446: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None
    clone_313: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_852: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_313, [32, -1, 80]);  clone_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_89: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_44, view_852);  div_44 = view_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_855: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_89, [1, 32, 128, 80]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_449: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_855, [0, 2, 1, 3]);  view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_316: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_449, memory_format = torch.contiguous_format);  permute_449 = None
    view_856: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_316, [1, 128, 2560]);  clone_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_857: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_856, [128, 2560]);  view_856 = None
    permute_450: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg591_1, [1, 0]);  arg591_1 = None
    addmm_225: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg592_1, view_857, permute_450);  arg592_1 = view_857 = permute_450 = None
    view_858: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_225, [1, 128, 2560]);  addmm_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_253: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_249, view_858);  add_249 = view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_69 = torch.ops.aten.var_mean.correction(add_253, [2], correction = 0, keepdim = True)
    getitem_138: "f32[1, 128, 1]" = var_mean_69[0]
    getitem_139: "f32[1, 128, 1]" = var_mean_69[1];  var_mean_69 = None
    sub_114: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_253, getitem_139);  getitem_139 = None
    add_254: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
    rsqrt_69: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
    mul_254: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_69);  sub_114 = rsqrt_69 = None
    mul_255: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_254, arg593_1);  mul_254 = arg593_1 = None
    add_255: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_255, arg594_1);  mul_255 = arg594_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_859: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_255, [128, 2560]);  add_255 = None
    permute_451: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg595_1, [1, 0]);  arg595_1 = None
    addmm_226: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg596_1, view_859, permute_451);  arg596_1 = view_859 = permute_451 = None
    view_860: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_226, [1, 128, 2560]);  addmm_226 = None
    mul_256: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_860, 0.11180339887498948);  view_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_867: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_256, [1, 128, 32, 80]);  mul_256 = None
    permute_456: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_867, [0, 2, 1, 3]);  view_867 = None
    clone_320: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_868: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_320, [32, -1, 80]);  clone_320 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_6: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_868, 0);  view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_861: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_452: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg597_1, [1, 0]);  arg597_1 = None
    addmm_227: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg598_1, view_861, permute_452);  arg598_1 = view_861 = permute_452 = None
    view_862: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_227, [1, 128, 2560]);  addmm_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_863: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_862, [1, -1, 32, 80]);  view_862 = None
    permute_453: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_863, [0, 2, 1, 3]);  view_863 = None
    clone_318: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_869: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_318, [32, -1, 80]);  clone_318 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_7: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_869, 0);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_864: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_454: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg599_1, [1, 0]);  arg599_1 = None
    addmm_228: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg600_1, view_864, permute_454);  arg600_1 = view_864 = permute_454 = None
    view_865: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_228, [1, 128, 2560]);  addmm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_866: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_865, [1, -1, 32, 80]);  view_865 = None
    permute_455: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_866, [0, 2, 1, 3]);  view_866 = None
    clone_319: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_870: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_319, [32, -1, 80]);  clone_319 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_8: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_870, 0);  view_870 = None
    _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
    getitem_158: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_2[0];  _scaled_dot_product_flash_attention_default_2 = None
    squeeze_dim_2: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_158, 0);  getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_871: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_2, [1, 32, 128, 80]);  squeeze_dim_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_458: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_871, [0, 2, 1, 3]);  view_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_322: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_872: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_322, [1, 128, 2560]);  clone_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_873: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_872, [128, 2560]);  view_872 = None
    permute_459: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg601_1, [1, 0]);  arg601_1 = None
    addmm_229: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg602_1, view_873, permute_459);  arg602_1 = view_873 = permute_459 = None
    view_874: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_229, [1, 128, 2560]);  addmm_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_256: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_253, view_874);  add_253 = view_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_70 = torch.ops.aten.var_mean.correction(add_256, [2], correction = 0, keepdim = True)
    getitem_140: "f32[1, 128, 1]" = var_mean_70[0]
    getitem_141: "f32[1, 128, 1]" = var_mean_70[1];  var_mean_70 = None
    sub_116: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_256, getitem_141);  getitem_141 = None
    add_257: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
    rsqrt_70: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
    mul_257: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_70);  sub_116 = rsqrt_70 = None
    mul_258: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_257, arg603_1);  mul_257 = arg603_1 = None
    add_258: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_258, arg604_1);  mul_258 = arg604_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_875: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_258, [128, 2560]);  add_258 = None
    permute_460: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg605_1, [1, 0]);  arg605_1 = None
    addmm_230: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg606_1, view_875, permute_460);  arg606_1 = view_875 = permute_460 = None
    view_876: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_230, [1, 128, 10240]);  addmm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_259: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_876, 0.5)
    mul_260: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_876, 0.7071067811865476);  view_876 = None
    erf_23: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_260);  mul_260 = None
    add_259: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_261: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_259, add_259);  mul_259 = add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_877: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_261, [128, 10240]);  mul_261 = None
    permute_461: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
    addmm_231: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg608_1, view_877, permute_461);  arg608_1 = view_877 = permute_461 = None
    view_878: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_231, [1, 128, 2560]);  addmm_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_260: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_256, view_878);  add_256 = view_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_71 = torch.ops.aten.var_mean.correction(add_260, [2], correction = 0, keepdim = True)
    getitem_142: "f32[1, 128, 1]" = var_mean_71[0]
    getitem_143: "f32[1, 128, 1]" = var_mean_71[1];  var_mean_71 = None
    sub_117: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_260, getitem_143);  getitem_143 = None
    add_261: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_71: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    mul_262: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_71);  sub_117 = rsqrt_71 = None
    mul_263: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_262, arg609_1);  mul_262 = arg609_1 = None
    add_262: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_263, arg610_1);  mul_263 = arg610_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_879: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_262, [128, 2560])
    permute_462: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg611_1, [1, 0]);  arg611_1 = None
    addmm_232: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg612_1, view_879, permute_462);  arg612_1 = view_879 = permute_462 = None
    view_880: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_232, [1, 128, 2560]);  addmm_232 = None
    mul_264: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_880, 0.11180339887498948);  view_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_887: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_264, [1, 128, 32, 80]);  mul_264 = None
    permute_467: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_887, [0, 2, 1, 3]);  view_887 = None
    clone_328: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_888: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_328, [32, -1, 80]);  clone_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_881: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_262, [128, 2560])
    permute_463: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg613_1, [1, 0]);  arg613_1 = None
    addmm_233: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg614_1, view_881, permute_463);  arg614_1 = view_881 = permute_463 = None
    view_882: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_233, [1, 128, 2560]);  addmm_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_883: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_882, [1, -1, 32, 80]);  view_882 = None
    permute_464: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_883, [0, 2, 1, 3]);  view_883 = None
    clone_326: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_464, memory_format = torch.contiguous_format);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_889: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_326, [32, -1, 80]);  clone_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_468: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_889, [0, 2, 1]);  view_889 = None
    bmm_92: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_888, permute_468);  view_888 = permute_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_891: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_92, [1, 32, 128, 128]);  bmm_92 = None
    add_263: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_891, expand_1);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_892: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_263, [32, 128, 128]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_46: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_892, [-1], True)
    sub_118: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_892, amax_46);  view_892 = amax_46 = None
    exp_46: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_118);  sub_118 = None
    sum_47: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
    div_46: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_884: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_262, [128, 2560]);  add_262 = None
    permute_465: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg615_1, [1, 0]);  arg615_1 = None
    addmm_234: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg616_1, view_884, permute_465);  arg616_1 = view_884 = permute_465 = None
    view_885: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_234, [1, 128, 2560]);  addmm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_886: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_885, [1, -1, 32, 80]);  view_885 = None
    permute_466: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None
    clone_327: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_890: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_327, [32, -1, 80]);  clone_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_93: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_46, view_890);  div_46 = view_890 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_893: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_93, [1, 32, 128, 80]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_469: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_330: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_469, memory_format = torch.contiguous_format);  permute_469 = None
    view_894: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_330, [1, 128, 2560]);  clone_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_895: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_894, [128, 2560]);  view_894 = None
    permute_470: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg617_1, [1, 0]);  arg617_1 = None
    addmm_235: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg618_1, view_895, permute_470);  arg618_1 = view_895 = permute_470 = None
    view_896: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_235, [1, 128, 2560]);  addmm_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_264: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_260, view_896);  add_260 = view_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_72 = torch.ops.aten.var_mean.correction(add_264, [2], correction = 0, keepdim = True)
    getitem_144: "f32[1, 128, 1]" = var_mean_72[0]
    getitem_145: "f32[1, 128, 1]" = var_mean_72[1];  var_mean_72 = None
    sub_119: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_264, getitem_145);  getitem_145 = None
    add_265: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
    rsqrt_72: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
    mul_265: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_72);  sub_119 = rsqrt_72 = None
    mul_266: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_265, arg619_1);  mul_265 = arg619_1 = None
    add_266: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_266, arg620_1);  mul_266 = arg620_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_897: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_266, [128, 2560]);  add_266 = None
    permute_471: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg621_1, [1, 0]);  arg621_1 = None
    addmm_236: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg622_1, view_897, permute_471);  arg622_1 = view_897 = permute_471 = None
    view_898: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_236, [1, 128, 2560]);  addmm_236 = None
    mul_267: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_898, 0.11180339887498948);  view_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_905: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_267, [1, 128, 32, 80]);  mul_267 = None
    permute_476: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_905, [0, 2, 1, 3]);  view_905 = None
    clone_334: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_476, memory_format = torch.contiguous_format);  permute_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_906: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_334, [32, -1, 80]);  clone_334 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_3: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_906, 0);  view_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_899: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_472: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg623_1, [1, 0]);  arg623_1 = None
    addmm_237: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg624_1, view_899, permute_472);  arg624_1 = view_899 = permute_472 = None
    view_900: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_237, [1, 128, 2560]);  addmm_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_901: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_900, [1, -1, 32, 80]);  view_900 = None
    permute_473: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_901, [0, 2, 1, 3]);  view_901 = None
    clone_332: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_907: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_332, [32, -1, 80]);  clone_332 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_4: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_907, 0);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_902: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_474: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
    addmm_238: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg626_1, view_902, permute_474);  arg626_1 = view_902 = permute_474 = None
    view_903: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_238, [1, 128, 2560]);  addmm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_904: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_903, [1, -1, 32, 80]);  view_903 = None
    permute_475: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_904, [0, 2, 1, 3]);  view_904 = None
    clone_333: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_475, memory_format = torch.contiguous_format);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_908: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_333, [32, -1, 80]);  clone_333 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_5: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_908, 0);  view_908 = None
    _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
    getitem_157: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default_1[0];  _scaled_dot_product_flash_attention_default_1 = None
    squeeze_dim_1: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_157, 0);  getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_909: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim_1, [1, 32, 128, 80]);  squeeze_dim_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_478: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_909, [0, 2, 1, 3]);  view_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_336: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
    view_910: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_336, [1, 128, 2560]);  clone_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_911: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_910, [128, 2560]);  view_910 = None
    permute_479: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg627_1, [1, 0]);  arg627_1 = None
    addmm_239: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg628_1, view_911, permute_479);  arg628_1 = view_911 = permute_479 = None
    view_912: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_239, [1, 128, 2560]);  addmm_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_267: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_264, view_912);  add_264 = view_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_73 = torch.ops.aten.var_mean.correction(add_267, [2], correction = 0, keepdim = True)
    getitem_146: "f32[1, 128, 1]" = var_mean_73[0]
    getitem_147: "f32[1, 128, 1]" = var_mean_73[1];  var_mean_73 = None
    sub_121: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_267, getitem_147);  getitem_147 = None
    add_268: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
    rsqrt_73: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
    mul_268: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_73);  sub_121 = rsqrt_73 = None
    mul_269: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_268, arg629_1);  mul_268 = arg629_1 = None
    add_269: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_269, arg630_1);  mul_269 = arg630_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_913: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_269, [128, 2560]);  add_269 = None
    permute_480: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg631_1, [1, 0]);  arg631_1 = None
    addmm_240: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg632_1, view_913, permute_480);  arg632_1 = view_913 = permute_480 = None
    view_914: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_240, [1, 128, 10240]);  addmm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_270: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_914, 0.5)
    mul_271: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_914, 0.7071067811865476);  view_914 = None
    erf_24: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_271);  mul_271 = None
    add_270: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_272: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_270, add_270);  mul_270 = add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_915: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_272, [128, 10240]);  mul_272 = None
    permute_481: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg633_1, [1, 0]);  arg633_1 = None
    addmm_241: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg634_1, view_915, permute_481);  arg634_1 = view_915 = permute_481 = None
    view_916: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_241, [1, 128, 2560]);  addmm_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_271: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_267, view_916);  add_267 = view_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_74 = torch.ops.aten.var_mean.correction(add_271, [2], correction = 0, keepdim = True)
    getitem_148: "f32[1, 128, 1]" = var_mean_74[0]
    getitem_149: "f32[1, 128, 1]" = var_mean_74[1];  var_mean_74 = None
    sub_122: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_271, getitem_149);  getitem_149 = None
    add_272: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_74: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    mul_273: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_74);  sub_122 = rsqrt_74 = None
    mul_274: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_273, arg635_1);  mul_273 = arg635_1 = None
    add_273: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_274, arg636_1);  mul_274 = arg636_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_917: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_273, [128, 2560])
    permute_482: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg637_1, [1, 0]);  arg637_1 = None
    addmm_242: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg638_1, view_917, permute_482);  arg638_1 = view_917 = permute_482 = None
    view_918: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_242, [1, 128, 2560]);  addmm_242 = None
    mul_275: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_918, 0.11180339887498948);  view_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_925: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_275, [1, 128, 32, 80]);  mul_275 = None
    permute_487: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_925, [0, 2, 1, 3]);  view_925 = None
    clone_342: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_926: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_342, [32, -1, 80]);  clone_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_919: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_273, [128, 2560])
    permute_483: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
    addmm_243: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg640_1, view_919, permute_483);  arg640_1 = view_919 = permute_483 = None
    view_920: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_243, [1, 128, 2560]);  addmm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_921: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_920, [1, -1, 32, 80]);  view_920 = None
    permute_484: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_921, [0, 2, 1, 3]);  view_921 = None
    clone_340: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_927: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_340, [32, -1, 80]);  clone_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_488: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_927, [0, 2, 1]);  view_927 = None
    bmm_96: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_926, permute_488);  view_926 = permute_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_929: "f32[1, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_96, [1, 32, 128, 128]);  bmm_96 = None
    add_274: "f32[1, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_929, expand_1);  view_929 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_930: "f32[32, 128, 128]" = torch.ops.aten.reshape.default(add_274, [32, 128, 128]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_48: "f32[32, 128, 1]" = torch.ops.aten.amax.default(view_930, [-1], True)
    sub_123: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(view_930, amax_48);  view_930 = amax_48 = None
    exp_48: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_123);  sub_123 = None
    sum_49: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_48, [-1], True)
    div_48: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp_48, sum_49);  exp_48 = sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_922: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_273, [128, 2560]);  add_273 = None
    permute_485: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg641_1, [1, 0]);  arg641_1 = None
    addmm_244: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg642_1, view_922, permute_485);  arg642_1 = view_922 = permute_485 = None
    view_923: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_244, [1, 128, 2560]);  addmm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_924: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_923, [1, -1, 32, 80]);  view_923 = None
    permute_486: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_924, [0, 2, 1, 3]);  view_924 = None
    clone_341: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_928: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_341, [32, -1, 80]);  clone_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_97: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(div_48, view_928);  div_48 = view_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_931: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_97, [1, 32, 128, 80]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_489: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_931, [0, 2, 1, 3]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_344: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    view_932: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_344, [1, 128, 2560]);  clone_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_933: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_932, [128, 2560]);  view_932 = None
    permute_490: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
    addmm_245: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg644_1, view_933, permute_490);  arg644_1 = view_933 = permute_490 = None
    view_934: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_245, [1, 128, 2560]);  addmm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    add_275: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_271, view_934);  add_271 = view_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_75 = torch.ops.aten.var_mean.correction(add_275, [2], correction = 0, keepdim = True)
    getitem_150: "f32[1, 128, 1]" = var_mean_75[0]
    getitem_151: "f32[1, 128, 1]" = var_mean_75[1];  var_mean_75 = None
    sub_124: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_275, getitem_151);  getitem_151 = None
    add_276: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
    rsqrt_75: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    mul_276: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_75);  sub_124 = rsqrt_75 = None
    mul_277: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_276, arg645_1);  mul_276 = arg645_1 = None
    add_277: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_277, arg646_1);  mul_277 = arg646_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_935: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_277, [128, 2560]);  add_277 = None
    permute_491: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg647_1, [1, 0]);  arg647_1 = None
    addmm_246: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg648_1, view_935, permute_491);  arg648_1 = view_935 = permute_491 = None
    view_936: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_246, [1, 128, 2560]);  addmm_246 = None
    mul_278: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_936, 0.11180339887498948);  view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_943: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(mul_278, [1, 128, 32, 80]);  mul_278 = None
    permute_496: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_943, [0, 2, 1, 3]);  view_943 = None
    clone_348: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_944: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_348, [32, -1, 80]);  clone_348 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_944, 0);  view_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_937: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_492: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg649_1, [1, 0]);  arg649_1 = None
    addmm_247: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg650_1, view_937, permute_492);  arg650_1 = view_937 = permute_492 = None
    view_938: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_247, [1, 128, 2560]);  addmm_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_939: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_938, [1, -1, 32, 80]);  view_938 = None
    permute_493: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_939, [0, 2, 1, 3]);  view_939 = None
    clone_346: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_493, memory_format = torch.contiguous_format);  permute_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_945: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_346, [32, -1, 80]);  clone_346 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_1: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_945, 0);  view_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_940: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_16, [128, 2560])
    permute_494: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg651_1, [1, 0]);  arg651_1 = None
    addmm_248: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg652_1, view_940, permute_494);  arg652_1 = view_940 = permute_494 = None
    view_941: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_248, [1, 128, 2560]);  addmm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_942: "f32[1, 128, 32, 80]" = torch.ops.aten.reshape.default(view_941, [1, -1, 32, 80]);  view_941 = None
    permute_495: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None
    clone_347: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_946: "f32[32, 128, 80]" = torch.ops.aten.reshape.default(clone_347, [32, -1, 80]);  clone_347 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_2: "f32[1, 32, 128, 80]" = torch.ops.aten.unsqueeze.default(view_946, 0);  view_946 = None
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem_156: "f32[1, 32, 128, 80]" = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
    squeeze_dim: "f32[32, 128, 80]" = torch.ops.aten.squeeze.dim(getitem_156, 0);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_947: "f32[1, 32, 128, 80]" = torch.ops.aten.reshape.default(squeeze_dim, [1, 32, 128, 80]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_498: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_947, [0, 2, 1, 3]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_350: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
    view_948: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(clone_350, [1, 128, 2560]);  clone_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_949: "f32[128, 2560]" = torch.ops.aten.reshape.default(view_948, [128, 2560]);  view_948 = None
    permute_499: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg653_1, [1, 0]);  arg653_1 = None
    addmm_249: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg654_1, view_949, permute_499);  arg654_1 = view_949 = permute_499 = None
    view_950: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_249, [1, 128, 2560]);  addmm_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    add_278: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_275, view_950);  add_275 = view_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_76 = torch.ops.aten.var_mean.correction(add_278, [2], correction = 0, keepdim = True)
    getitem_152: "f32[1, 128, 1]" = var_mean_76[0]
    getitem_153: "f32[1, 128, 1]" = var_mean_76[1];  var_mean_76 = None
    sub_126: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_278, getitem_153);  getitem_153 = None
    add_279: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
    rsqrt_76: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
    mul_279: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_76);  sub_126 = rsqrt_76 = None
    mul_280: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_279, arg655_1);  mul_279 = arg655_1 = None
    add_280: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_280, arg656_1);  mul_280 = arg656_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_951: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_280, [128, 2560]);  add_280 = None
    permute_500: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg657_1, [1, 0]);  arg657_1 = None
    addmm_250: "f32[128, 10240]" = torch.ops.aten.addmm.default(arg658_1, view_951, permute_500);  arg658_1 = view_951 = permute_500 = None
    view_952: "f32[1, 128, 10240]" = torch.ops.aten.reshape.default(addmm_250, [1, 128, 10240]);  addmm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_281: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_952, 0.5)
    mul_282: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_952, 0.7071067811865476);  view_952 = None
    erf_25: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_282);  mul_282 = None
    add_281: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_283: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_281, add_281);  mul_281 = add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    view_953: "f32[128, 10240]" = torch.ops.aten.reshape.default(mul_283, [128, 10240]);  mul_283 = None
    permute_501: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg659_1, [1, 0]);  arg659_1 = None
    addmm_251: "f32[128, 2560]" = torch.ops.aten.addmm.default(arg660_1, view_953, permute_501);  arg660_1 = view_953 = permute_501 = None
    view_954: "f32[1, 128, 2560]" = torch.ops.aten.reshape.default(addmm_251, [1, 128, 2560]);  addmm_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    add_282: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_278, view_954);  add_278 = view_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1079, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_77 = torch.ops.aten.var_mean.correction(add_282, [2], correction = 0, keepdim = True)
    getitem_154: "f32[1, 128, 1]" = var_mean_77[0]
    getitem_155: "f32[1, 128, 1]" = var_mean_77[1];  var_mean_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1363, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_958: "i64[128]" = torch.ops.aten.reshape.default(arg665_1, [-1]);  arg665_1 = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_958, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1079, code: hidden_states = self.layer_norm(hidden_states)
    sub_127: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_282, getitem_155);  add_282 = getitem_155 = None
    add_283: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
    rsqrt_77: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
    mul_284: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_77);  sub_127 = rsqrt_77 = None
    mul_285: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_284, arg661_1);  mul_284 = arg661_1 = None
    add_284: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_285, arg662_1);  mul_285 = arg662_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1358, code: lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    view_955: "f32[128, 2560]" = torch.ops.aten.reshape.default(add_284, [128, 2560]);  add_284 = None
    permute_502: "f32[2560, 8008]" = torch.ops.aten.permute.default(arg663_1, [1, 0]);  arg663_1 = None
    mm: "f32[128, 8008]" = torch.ops.aten.mm.default(view_955, permute_502);  view_955 = permute_502 = None
    view_956: "f32[1, 128, 8008]" = torch.ops.aten.reshape.default(mm, [1, 128, 8008]);  mm = None
    add_285: "f32[1, 128, 8008]" = torch.ops.aten.add.Tensor(view_956, arg664_1);  view_956 = arg664_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1363, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_957: "f32[128, 8008]" = torch.ops.aten.reshape.default(add_285, [-1, 8008])
    amax_50: "f32[128, 1]" = torch.ops.aten.amax.default(view_957, [1], True)
    sub_128: "f32[128, 8008]" = torch.ops.aten.sub.Tensor(view_957, amax_50);  view_957 = amax_50 = None
    exp_50: "f32[128, 8008]" = torch.ops.aten.exp.default(sub_128)
    sum_51: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_50, [1], True);  exp_50 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_51);  sum_51 = None
    sub_129: "f32[128, 8008]" = torch.ops.aten.sub.Tensor(sub_128, log);  sub_128 = log = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_958, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "i64[128]" = torch.ops.aten.where.self(ne, view_958, full_default_2);  ne = full_default_2 = None
    unsqueeze_4: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_129, 1, unsqueeze_4);  sub_129 = unsqueeze_4 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_2: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
    sum_53: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_958, -100);  view_958 = None
    sum_52: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_52, torch.float32);  sum_52 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(sum_53, convert_element_type);  sum_53 = convert_element_type = None
    return (div_50, add_285, add_16)
    