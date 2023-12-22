from __future__ import annotations



def forward(self, arg0_1: "f32[1, 576, 768]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[1, 1, 768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768, 3, 16, 16]", arg79_1: "f32[768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[2304, 768]", arg83_1: "f32[2304]", arg84_1: "f32[16, 16]", arg85_1: "f32[16]", arg86_1: "f32[16, 16]", arg87_1: "f32[16]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[768]", arg92_1: "f32[3072, 768]", arg93_1: "f32[3072]", arg94_1: "f32[768, 3072]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[2304, 768]", arg99_1: "f32[2304]", arg100_1: "f32[16, 16]", arg101_1: "f32[16]", arg102_1: "f32[16, 16]", arg103_1: "f32[16]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[3072, 768]", arg109_1: "f32[3072]", arg110_1: "f32[768, 3072]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[2304, 768]", arg115_1: "f32[2304]", arg116_1: "f32[16, 16]", arg117_1: "f32[16]", arg118_1: "f32[16, 16]", arg119_1: "f32[16]", arg120_1: "f32[768, 768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[3072, 768]", arg125_1: "f32[3072]", arg126_1: "f32[768, 3072]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[2304, 768]", arg131_1: "f32[2304]", arg132_1: "f32[16, 16]", arg133_1: "f32[16]", arg134_1: "f32[16, 16]", arg135_1: "f32[16]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[3072, 768]", arg141_1: "f32[3072]", arg142_1: "f32[768, 3072]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[2304, 768]", arg147_1: "f32[2304]", arg148_1: "f32[16, 16]", arg149_1: "f32[16]", arg150_1: "f32[16, 16]", arg151_1: "f32[16]", arg152_1: "f32[768, 768]", arg153_1: "f32[768]", arg154_1: "f32[768]", arg155_1: "f32[768]", arg156_1: "f32[3072, 768]", arg157_1: "f32[3072]", arg158_1: "f32[768, 3072]", arg159_1: "f32[768]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[2304, 768]", arg163_1: "f32[2304]", arg164_1: "f32[16, 16]", arg165_1: "f32[16]", arg166_1: "f32[16, 16]", arg167_1: "f32[16]", arg168_1: "f32[768, 768]", arg169_1: "f32[768]", arg170_1: "f32[768]", arg171_1: "f32[768]", arg172_1: "f32[3072, 768]", arg173_1: "f32[3072]", arg174_1: "f32[768, 3072]", arg175_1: "f32[768]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[2304, 768]", arg179_1: "f32[2304]", arg180_1: "f32[16, 16]", arg181_1: "f32[16]", arg182_1: "f32[16, 16]", arg183_1: "f32[16]", arg184_1: "f32[768, 768]", arg185_1: "f32[768]", arg186_1: "f32[768]", arg187_1: "f32[768]", arg188_1: "f32[3072, 768]", arg189_1: "f32[3072]", arg190_1: "f32[768, 3072]", arg191_1: "f32[768]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[2304, 768]", arg195_1: "f32[2304]", arg196_1: "f32[16, 16]", arg197_1: "f32[16]", arg198_1: "f32[16, 16]", arg199_1: "f32[16]", arg200_1: "f32[768, 768]", arg201_1: "f32[768]", arg202_1: "f32[768]", arg203_1: "f32[768]", arg204_1: "f32[3072, 768]", arg205_1: "f32[3072]", arg206_1: "f32[768, 3072]", arg207_1: "f32[768]", arg208_1: "f32[768]", arg209_1: "f32[768]", arg210_1: "f32[2304, 768]", arg211_1: "f32[2304]", arg212_1: "f32[16, 16]", arg213_1: "f32[16]", arg214_1: "f32[16, 16]", arg215_1: "f32[16]", arg216_1: "f32[768, 768]", arg217_1: "f32[768]", arg218_1: "f32[768]", arg219_1: "f32[768]", arg220_1: "f32[3072, 768]", arg221_1: "f32[3072]", arg222_1: "f32[768, 3072]", arg223_1: "f32[768]", arg224_1: "f32[768]", arg225_1: "f32[768]", arg226_1: "f32[2304, 768]", arg227_1: "f32[2304]", arg228_1: "f32[16, 16]", arg229_1: "f32[16]", arg230_1: "f32[16, 16]", arg231_1: "f32[16]", arg232_1: "f32[768, 768]", arg233_1: "f32[768]", arg234_1: "f32[768]", arg235_1: "f32[768]", arg236_1: "f32[3072, 768]", arg237_1: "f32[3072]", arg238_1: "f32[768, 3072]", arg239_1: "f32[768]", arg240_1: "f32[768]", arg241_1: "f32[768]", arg242_1: "f32[2304, 768]", arg243_1: "f32[2304]", arg244_1: "f32[16, 16]", arg245_1: "f32[16]", arg246_1: "f32[16, 16]", arg247_1: "f32[16]", arg248_1: "f32[768, 768]", arg249_1: "f32[768]", arg250_1: "f32[768]", arg251_1: "f32[768]", arg252_1: "f32[3072, 768]", arg253_1: "f32[3072]", arg254_1: "f32[768, 3072]", arg255_1: "f32[768]", arg256_1: "f32[768]", arg257_1: "f32[768]", arg258_1: "f32[2304, 768]", arg259_1: "f32[2304]", arg260_1: "f32[16, 16]", arg261_1: "f32[16]", arg262_1: "f32[16, 16]", arg263_1: "f32[16]", arg264_1: "f32[768, 768]", arg265_1: "f32[768]", arg266_1: "f32[768]", arg267_1: "f32[768]", arg268_1: "f32[3072, 768]", arg269_1: "f32[3072]", arg270_1: "f32[768, 3072]", arg271_1: "f32[768]", arg272_1: "f32[768]", arg273_1: "f32[768]", arg274_1: "f32[2304, 768]", arg275_1: "f32[2304]", arg276_1: "f32[16, 16]", arg277_1: "f32[16]", arg278_1: "f32[16, 16]", arg279_1: "f32[16]", arg280_1: "f32[768, 768]", arg281_1: "f32[768]", arg282_1: "f32[768]", arg283_1: "f32[768]", arg284_1: "f32[3072, 768]", arg285_1: "f32[3072]", arg286_1: "f32[768, 3072]", arg287_1: "f32[768]", arg288_1: "f32[768]", arg289_1: "f32[768]", arg290_1: "f32[2304, 768]", arg291_1: "f32[2304]", arg292_1: "f32[16, 16]", arg293_1: "f32[16]", arg294_1: "f32[16, 16]", arg295_1: "f32[16]", arg296_1: "f32[768, 768]", arg297_1: "f32[768]", arg298_1: "f32[768]", arg299_1: "f32[768]", arg300_1: "f32[3072, 768]", arg301_1: "f32[3072]", arg302_1: "f32[768, 3072]", arg303_1: "f32[768]", arg304_1: "f32[768]", arg305_1: "f32[768]", arg306_1: "f32[2304, 768]", arg307_1: "f32[2304]", arg308_1: "f32[16, 16]", arg309_1: "f32[16]", arg310_1: "f32[16, 16]", arg311_1: "f32[16]", arg312_1: "f32[768, 768]", arg313_1: "f32[768]", arg314_1: "f32[768]", arg315_1: "f32[768]", arg316_1: "f32[3072, 768]", arg317_1: "f32[3072]", arg318_1: "f32[768, 3072]", arg319_1: "f32[768]", arg320_1: "f32[768]", arg321_1: "f32[768]", arg322_1: "f32[2304, 768]", arg323_1: "f32[2304]", arg324_1: "f32[16, 16]", arg325_1: "f32[16]", arg326_1: "f32[16, 16]", arg327_1: "f32[16]", arg328_1: "f32[768, 768]", arg329_1: "f32[768]", arg330_1: "f32[768]", arg331_1: "f32[768]", arg332_1: "f32[3072, 768]", arg333_1: "f32[3072]", arg334_1: "f32[768, 3072]", arg335_1: "f32[768]", arg336_1: "f32[768]", arg337_1: "f32[768]", arg338_1: "f32[2304, 768]", arg339_1: "f32[2304]", arg340_1: "f32[16, 16]", arg341_1: "f32[16]", arg342_1: "f32[16, 16]", arg343_1: "f32[16]", arg344_1: "f32[768, 768]", arg345_1: "f32[768]", arg346_1: "f32[768]", arg347_1: "f32[768]", arg348_1: "f32[3072, 768]", arg349_1: "f32[3072]", arg350_1: "f32[768, 3072]", arg351_1: "f32[768]", arg352_1: "f32[768]", arg353_1: "f32[768]", arg354_1: "f32[2304, 768]", arg355_1: "f32[2304]", arg356_1: "f32[16, 16]", arg357_1: "f32[16]", arg358_1: "f32[16, 16]", arg359_1: "f32[16]", arg360_1: "f32[768, 768]", arg361_1: "f32[768]", arg362_1: "f32[768]", arg363_1: "f32[768]", arg364_1: "f32[3072, 768]", arg365_1: "f32[3072]", arg366_1: "f32[768, 3072]", arg367_1: "f32[768]", arg368_1: "f32[768]", arg369_1: "f32[768]", arg370_1: "f32[2304, 768]", arg371_1: "f32[2304]", arg372_1: "f32[16, 16]", arg373_1: "f32[16]", arg374_1: "f32[16, 16]", arg375_1: "f32[16]", arg376_1: "f32[768, 768]", arg377_1: "f32[768]", arg378_1: "f32[768]", arg379_1: "f32[768]", arg380_1: "f32[3072, 768]", arg381_1: "f32[3072]", arg382_1: "f32[768, 3072]", arg383_1: "f32[768]", arg384_1: "f32[768]", arg385_1: "f32[768]", arg386_1: "f32[2304, 768]", arg387_1: "f32[2304]", arg388_1: "f32[16, 16]", arg389_1: "f32[16]", arg390_1: "f32[16, 16]", arg391_1: "f32[16]", arg392_1: "f32[768, 768]", arg393_1: "f32[768]", arg394_1: "f32[768]", arg395_1: "f32[768]", arg396_1: "f32[3072, 768]", arg397_1: "f32[3072]", arg398_1: "f32[768, 3072]", arg399_1: "f32[768]", arg400_1: "f32[768]", arg401_1: "f32[768]", arg402_1: "f32[2304, 768]", arg403_1: "f32[2304]", arg404_1: "f32[16, 16]", arg405_1: "f32[16]", arg406_1: "f32[16, 16]", arg407_1: "f32[16]", arg408_1: "f32[768, 768]", arg409_1: "f32[768]", arg410_1: "f32[768]", arg411_1: "f32[768]", arg412_1: "f32[3072, 768]", arg413_1: "f32[3072]", arg414_1: "f32[768, 3072]", arg415_1: "f32[768]", arg416_1: "f32[768]", arg417_1: "f32[768]", arg418_1: "f32[2304, 768]", arg419_1: "f32[2304]", arg420_1: "f32[16, 16]", arg421_1: "f32[16]", arg422_1: "f32[16, 16]", arg423_1: "f32[16]", arg424_1: "f32[768, 768]", arg425_1: "f32[768]", arg426_1: "f32[768]", arg427_1: "f32[768]", arg428_1: "f32[3072, 768]", arg429_1: "f32[3072]", arg430_1: "f32[768, 3072]", arg431_1: "f32[768]", arg432_1: "f32[768]", arg433_1: "f32[768]", arg434_1: "f32[2304, 768]", arg435_1: "f32[2304]", arg436_1: "f32[16, 16]", arg437_1: "f32[16]", arg438_1: "f32[16, 16]", arg439_1: "f32[16]", arg440_1: "f32[768, 768]", arg441_1: "f32[768]", arg442_1: "f32[768]", arg443_1: "f32[768]", arg444_1: "f32[3072, 768]", arg445_1: "f32[3072]", arg446_1: "f32[768, 3072]", arg447_1: "f32[768]", arg448_1: "f32[768]", arg449_1: "f32[768]", arg450_1: "f32[2304, 768]", arg451_1: "f32[2304]", arg452_1: "f32[16, 16]", arg453_1: "f32[16]", arg454_1: "f32[16, 16]", arg455_1: "f32[16]", arg456_1: "f32[768, 768]", arg457_1: "f32[768]", arg458_1: "f32[768]", arg459_1: "f32[768]", arg460_1: "f32[3072, 768]", arg461_1: "f32[3072]", arg462_1: "f32[768, 3072]", arg463_1: "f32[768]", arg464_1: "f32[768]", arg465_1: "f32[768]", arg466_1: "f32[2304, 768]", arg467_1: "f32[2304]", arg468_1: "f32[16, 16]", arg469_1: "f32[16]", arg470_1: "f32[16, 16]", arg471_1: "f32[16]", arg472_1: "f32[768, 768]", arg473_1: "f32[768]", arg474_1: "f32[768]", arg475_1: "f32[768]", arg476_1: "f32[3072, 768]", arg477_1: "f32[3072]", arg478_1: "f32[768, 3072]", arg479_1: "f32[768]", arg480_1: "f32[768]", arg481_1: "f32[768]", arg482_1: "f32[2304, 768]", arg483_1: "f32[2304]", arg484_1: "f32[16, 16]", arg485_1: "f32[16]", arg486_1: "f32[16, 16]", arg487_1: "f32[16]", arg488_1: "f32[768, 768]", arg489_1: "f32[768]", arg490_1: "f32[768]", arg491_1: "f32[768]", arg492_1: "f32[3072, 768]", arg493_1: "f32[3072]", arg494_1: "f32[768, 3072]", arg495_1: "f32[768]", arg496_1: "f32[768]", arg497_1: "f32[768]", arg498_1: "f32[2304, 768]", arg499_1: "f32[2304]", arg500_1: "f32[16, 16]", arg501_1: "f32[16]", arg502_1: "f32[16, 16]", arg503_1: "f32[16]", arg504_1: "f32[768, 768]", arg505_1: "f32[768]", arg506_1: "f32[768]", arg507_1: "f32[768]", arg508_1: "f32[3072, 768]", arg509_1: "f32[3072]", arg510_1: "f32[768, 3072]", arg511_1: "f32[768]", arg512_1: "f32[768]", arg513_1: "f32[768]", arg514_1: "f32[2304, 768]", arg515_1: "f32[2304]", arg516_1: "f32[16, 16]", arg517_1: "f32[16]", arg518_1: "f32[16, 16]", arg519_1: "f32[16]", arg520_1: "f32[768, 768]", arg521_1: "f32[768]", arg522_1: "f32[768]", arg523_1: "f32[768]", arg524_1: "f32[3072, 768]", arg525_1: "f32[3072]", arg526_1: "f32[768, 3072]", arg527_1: "f32[768]", arg528_1: "f32[768]", arg529_1: "f32[768]", arg530_1: "f32[2304, 768]", arg531_1: "f32[2304]", arg532_1: "f32[16, 16]", arg533_1: "f32[16]", arg534_1: "f32[16, 16]", arg535_1: "f32[16]", arg536_1: "f32[768, 768]", arg537_1: "f32[768]", arg538_1: "f32[768]", arg539_1: "f32[768]", arg540_1: "f32[3072, 768]", arg541_1: "f32[3072]", arg542_1: "f32[768, 3072]", arg543_1: "f32[768]", arg544_1: "f32[768]", arg545_1: "f32[768]", arg546_1: "f32[2304, 768]", arg547_1: "f32[2304]", arg548_1: "f32[16, 16]", arg549_1: "f32[16]", arg550_1: "f32[16, 16]", arg551_1: "f32[16]", arg552_1: "f32[768, 768]", arg553_1: "f32[768]", arg554_1: "f32[768]", arg555_1: "f32[768]", arg556_1: "f32[3072, 768]", arg557_1: "f32[3072]", arg558_1: "f32[768, 3072]", arg559_1: "f32[768]", arg560_1: "f32[768]", arg561_1: "f32[768]", arg562_1: "f32[2304, 768]", arg563_1: "f32[2304]", arg564_1: "f32[16, 16]", arg565_1: "f32[16]", arg566_1: "f32[16, 16]", arg567_1: "f32[16]", arg568_1: "f32[768, 768]", arg569_1: "f32[768]", arg570_1: "f32[768]", arg571_1: "f32[768]", arg572_1: "f32[3072, 768]", arg573_1: "f32[3072]", arg574_1: "f32[768, 3072]", arg575_1: "f32[768]", arg576_1: "f32[768]", arg577_1: "f32[768]", arg578_1: "f32[2304, 768]", arg579_1: "f32[2304]", arg580_1: "f32[16, 16]", arg581_1: "f32[16]", arg582_1: "f32[16, 16]", arg583_1: "f32[16]", arg584_1: "f32[768, 768]", arg585_1: "f32[768]", arg586_1: "f32[768]", arg587_1: "f32[768]", arg588_1: "f32[3072, 768]", arg589_1: "f32[3072]", arg590_1: "f32[768, 3072]", arg591_1: "f32[768]", arg592_1: "f32[768]", arg593_1: "f32[768]", arg594_1: "f32[2304, 768]", arg595_1: "f32[2304]", arg596_1: "f32[16, 16]", arg597_1: "f32[16]", arg598_1: "f32[16, 16]", arg599_1: "f32[16]", arg600_1: "f32[768, 768]", arg601_1: "f32[768]", arg602_1: "f32[768]", arg603_1: "f32[768]", arg604_1: "f32[3072, 768]", arg605_1: "f32[3072]", arg606_1: "f32[768, 3072]", arg607_1: "f32[768]", arg608_1: "f32[768]", arg609_1: "f32[768]", arg610_1: "f32[2304, 768]", arg611_1: "f32[2304]", arg612_1: "f32[16, 16]", arg613_1: "f32[16]", arg614_1: "f32[16, 16]", arg615_1: "f32[16]", arg616_1: "f32[768, 768]", arg617_1: "f32[768]", arg618_1: "f32[768]", arg619_1: "f32[768]", arg620_1: "f32[3072, 768]", arg621_1: "f32[3072]", arg622_1: "f32[768, 3072]", arg623_1: "f32[768]", arg624_1: "f32[768]", arg625_1: "f32[768]", arg626_1: "f32[2304, 768]", arg627_1: "f32[2304]", arg628_1: "f32[16, 16]", arg629_1: "f32[16]", arg630_1: "f32[16, 16]", arg631_1: "f32[16]", arg632_1: "f32[768, 768]", arg633_1: "f32[768]", arg634_1: "f32[768]", arg635_1: "f32[768]", arg636_1: "f32[3072, 768]", arg637_1: "f32[3072]", arg638_1: "f32[768, 3072]", arg639_1: "f32[768]", arg640_1: "f32[768]", arg641_1: "f32[768]", arg642_1: "f32[2304, 768]", arg643_1: "f32[2304]", arg644_1: "f32[16, 16]", arg645_1: "f32[16]", arg646_1: "f32[16, 16]", arg647_1: "f32[16]", arg648_1: "f32[768, 768]", arg649_1: "f32[768]", arg650_1: "f32[768]", arg651_1: "f32[768]", arg652_1: "f32[3072, 768]", arg653_1: "f32[3072]", arg654_1: "f32[768, 3072]", arg655_1: "f32[768]", arg656_1: "f32[768]", arg657_1: "f32[768]", arg658_1: "f32[768, 768]", arg659_1: "f32[768]", arg660_1: "f32[768, 768]", arg661_1: "f32[768]", arg662_1: "f32[768, 768]", arg663_1: "f32[768]", arg664_1: "f32[768, 768]", arg665_1: "f32[768]", arg666_1: "f32[768]", arg667_1: "f32[768]", arg668_1: "f32[3072, 768]", arg669_1: "f32[3072]", arg670_1: "f32[768, 3072]", arg671_1: "f32[768]", arg672_1: "f32[768]", arg673_1: "f32[768]", arg674_1: "f32[768, 768]", arg675_1: "f32[768]", arg676_1: "f32[768, 768]", arg677_1: "f32[768]", arg678_1: "f32[768, 768]", arg679_1: "f32[768]", arg680_1: "f32[768, 768]", arg681_1: "f32[768]", arg682_1: "f32[768]", arg683_1: "f32[768]", arg684_1: "f32[3072, 768]", arg685_1: "f32[3072]", arg686_1: "f32[768, 3072]", arg687_1: "f32[768]", arg688_1: "f32[768]", arg689_1: "f32[768]", arg690_1: "f32[1000, 768]", arg691_1: "f32[1000]", arg692_1: "f32[8, 3, 384, 384]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 24, 24]" = torch.ops.aten.convolution.default(arg692_1, arg78_1, arg79_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg692_1 = arg78_1 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 576]" = torch.ops.aten.view.default(convolution, [8, 768, 576]);  convolution = None
    permute: "f32[8, 576, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:341, code: x = x + self.pos_embed
    add: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(permute, arg0_1);  permute = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:342, code: x = self.pos_drop(x)
    clone: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_1: "f32[8, 576, 768]" = torch.ops.aten.clone.default(clone, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 576, 1]" = var_mean[0]
    getitem_1: "f32[8, 576, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
    mul: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul, arg80_1);  mul = arg80_1 = None
    add_2: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_1, arg81_1);  mul_1 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_1: "f32[4608, 768]" = torch.ops.aten.view.default(add_2, [4608, 768]);  add_2 = None
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg83_1, view_1, permute_1);  arg83_1 = view_1 = permute_1 = None
    view_2: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm, [8, 576, 2304]);  addmm = None
    view_3: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_2, [8, 576, 3, 16, 48]);  view_2 = None
    permute_2: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_2, 0, 0)
    mul_2: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select, 0.14433756729740643);  select = None
    select_1: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_2, 0, 1)
    select_2: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_2, 0, 2);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_3: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_1, [0, 1, 3, 2]);  select_1 = None
    expand: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_2, [8, 16, 576, 48]);  mul_2 = None
    clone_2: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_4: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_2, [128, 576, 48]);  clone_2 = None
    expand_1: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_3, [8, 16, 48, 576]);  permute_3 = None
    clone_3: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_5: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_3, [128, 48, 576]);  clone_3 = None
    bmm: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    view_6: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm, [8, 16, 576, 576]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_4: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_6, [0, 2, 3, 1]);  view_6 = None
    permute_5: "f32[16, 16]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    clone_4: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_7: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_4, [2654208, 16]);  clone_4 = None
    mm: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_7, permute_5);  view_7 = permute_5 = None
    view_8: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm, [8, 576, 576, 16]);  mm = None
    add_3: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_8, arg85_1);  view_8 = arg85_1 = None
    permute_6: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_3, [0, 3, 1, 2]);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_5: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    amax: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_5, [-1], True)
    sub_1: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_5, amax);  clone_5 = amax = None
    exp: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_7: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div, [0, 2, 3, 1]);  div = None
    permute_8: "f32[16, 16]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    clone_6: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_6, [2654208, 16]);  clone_6 = None
    mm_1: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_9, permute_8);  view_9 = permute_8 = None
    view_10: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_1, [8, 576, 576, 16]);  mm_1 = None
    add_4: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_10, arg87_1);  view_10 = arg87_1 = None
    permute_9: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_4, [0, 3, 1, 2]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_7: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_9);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_2: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_7, [8, 16, 576, 576]);  clone_7 = None
    clone_8: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_11: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_8, [128, 576, 576]);  clone_8 = None
    expand_3: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_2, [8, 16, 576, 48]);  select_2 = None
    clone_9: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_12: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_9, [128, 576, 48]);  clone_9 = None
    bmm_1: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
    view_13: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_1, [8, 16, 576, 48]);  bmm_1 = None
    permute_10: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    clone_10: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    view_14: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_10, [8, 576, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_15: "f32[4608, 768]" = torch.ops.aten.view.default(view_14, [4608, 768]);  view_14 = None
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_1: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg89_1, view_15, permute_11);  arg89_1 = view_15 = permute_11 = None
    view_16: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_1, [8, 576, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_11: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_3: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg1_1, clone_11);  arg1_1 = clone_11 = None
    add_5: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(clone, mul_3);  clone = mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_12: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_5, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 576, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 576, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_3);  clone_12 = getitem_3 = None
    mul_4: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg90_1);  mul_4 = arg90_1 = None
    add_7: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_5, arg91_1);  mul_5 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[4608, 768]" = torch.ops.aten.view.default(add_7, [4608, 768]);  add_7 = None
    permute_12: "f32[768, 3072]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_2: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg93_1, view_17, permute_12);  arg93_1 = view_17 = permute_12 = None
    view_18: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 576, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_6: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_8: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_13, [4608, 3072]);  clone_13 = None
    permute_13: "f32[3072, 768]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_3: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg95_1, view_19, permute_13);  arg95_1 = view_19 = permute_13 = None
    view_20: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_3, [8, 576, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_9: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg2_1, clone_14);  arg2_1 = clone_14 = None
    add_9: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_5, mul_9);  add_5 = mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_15: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_9, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 576, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 576, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_15, getitem_5);  clone_15 = getitem_5 = None
    mul_10: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_11: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg96_1);  mul_10 = arg96_1 = None
    add_11: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_11, arg97_1);  mul_11 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_21: "f32[4608, 768]" = torch.ops.aten.view.default(add_11, [4608, 768]);  add_11 = None
    permute_14: "f32[768, 2304]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_4: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg99_1, view_21, permute_14);  arg99_1 = view_21 = permute_14 = None
    view_22: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 576, 2304]);  addmm_4 = None
    view_23: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_22, [8, 576, 3, 16, 48]);  view_22 = None
    permute_15: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_3: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_15, 0, 0)
    mul_12: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_3, 0.14433756729740643);  select_3 = None
    select_4: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_15, 0, 1)
    select_5: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_15, 0, 2);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_16: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_4, [0, 1, 3, 2]);  select_4 = None
    expand_4: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_12, [8, 16, 576, 48]);  mul_12 = None
    clone_16: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_24: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_16, [128, 576, 48]);  clone_16 = None
    expand_5: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_16, [8, 16, 48, 576]);  permute_16 = None
    clone_17: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_25: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_17, [128, 48, 576]);  clone_17 = None
    bmm_2: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = view_25 = None
    view_26: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_2, [8, 16, 576, 576]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_17: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_26, [0, 2, 3, 1]);  view_26 = None
    permute_18: "f32[16, 16]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    clone_18: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_27: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_18, [2654208, 16]);  clone_18 = None
    mm_2: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_27, permute_18);  view_27 = permute_18 = None
    view_28: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_2, [8, 576, 576, 16]);  mm_2 = None
    add_12: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_28, arg101_1);  view_28 = arg101_1 = None
    permute_19: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_12, [0, 3, 1, 2]);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_19: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    amax_1: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_19, [-1], True)
    sub_4: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_19, amax_1);  clone_19 = amax_1 = None
    exp_1: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_20: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_1, [0, 2, 3, 1]);  div_1 = None
    permute_21: "f32[16, 16]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    clone_20: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_29: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_20, [2654208, 16]);  clone_20 = None
    mm_3: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_29, permute_21);  view_29 = permute_21 = None
    view_30: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_3, [8, 576, 576, 16]);  mm_3 = None
    add_13: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_30, arg103_1);  view_30 = arg103_1 = None
    permute_22: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_13, [0, 3, 1, 2]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_21: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_22);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_6: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_21, [8, 16, 576, 576]);  clone_21 = None
    clone_22: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_31: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_22, [128, 576, 576]);  clone_22 = None
    expand_7: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_5, [8, 16, 576, 48]);  select_5 = None
    clone_23: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_32: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_23, [128, 576, 48]);  clone_23 = None
    bmm_3: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
    view_33: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_3, [8, 16, 576, 48]);  bmm_3 = None
    permute_23: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    clone_24: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_34: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_24, [8, 576, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_35: "f32[4608, 768]" = torch.ops.aten.view.default(view_34, [4608, 768]);  view_34 = None
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_5: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg105_1, view_35, permute_24);  arg105_1 = view_35 = permute_24 = None
    view_36: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_5, [8, 576, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_25: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_13: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg3_1, clone_25);  arg3_1 = clone_25 = None
    add_14: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_9, mul_13);  add_9 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_26: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_26, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 576, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 576, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_5: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_26, getitem_7);  clone_26 = getitem_7 = None
    mul_14: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_15: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg106_1);  mul_14 = arg106_1 = None
    add_16: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_15, arg107_1);  mul_15 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[4608, 768]" = torch.ops.aten.view.default(add_16, [4608, 768]);  add_16 = None
    permute_25: "f32[768, 3072]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_6: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg109_1, view_37, permute_25);  arg109_1 = view_37 = permute_25 = None
    view_38: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 576, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_16: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_17: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_17: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_18: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_16, add_17);  mul_16 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_27: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_27, [4608, 3072]);  clone_27 = None
    permute_26: "f32[3072, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_7: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg111_1, view_39, permute_26);  arg111_1 = view_39 = permute_26 = None
    view_40: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_7, [8, 576, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_28: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_19: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg4_1, clone_28);  arg4_1 = clone_28 = None
    add_18: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_14, mul_19);  add_14 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_29: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_18, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 576, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 576, 1]" = var_mean_4[1];  var_mean_4 = None
    add_19: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_6: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_29, getitem_9);  clone_29 = getitem_9 = None
    mul_20: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_21: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg112_1);  mul_20 = arg112_1 = None
    add_20: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_21, arg113_1);  mul_21 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_41: "f32[4608, 768]" = torch.ops.aten.view.default(add_20, [4608, 768]);  add_20 = None
    permute_27: "f32[768, 2304]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_8: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg115_1, view_41, permute_27);  arg115_1 = view_41 = permute_27 = None
    view_42: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 576, 2304]);  addmm_8 = None
    view_43: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_42, [8, 576, 3, 16, 48]);  view_42 = None
    permute_28: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_6: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_28, 0, 0)
    mul_22: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_6, 0.14433756729740643);  select_6 = None
    select_7: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_28, 0, 1)
    select_8: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_28, 0, 2);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_29: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_7, [0, 1, 3, 2]);  select_7 = None
    expand_8: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_22, [8, 16, 576, 48]);  mul_22 = None
    clone_30: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_44: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_30, [128, 576, 48]);  clone_30 = None
    expand_9: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_29, [8, 16, 48, 576]);  permute_29 = None
    clone_31: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_45: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_31, [128, 48, 576]);  clone_31 = None
    bmm_4: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_44, view_45);  view_44 = view_45 = None
    view_46: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_4, [8, 16, 576, 576]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_30: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_46, [0, 2, 3, 1]);  view_46 = None
    permute_31: "f32[16, 16]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    clone_32: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_47: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_32, [2654208, 16]);  clone_32 = None
    mm_4: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_47, permute_31);  view_47 = permute_31 = None
    view_48: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_4, [8, 576, 576, 16]);  mm_4 = None
    add_21: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_48, arg117_1);  view_48 = arg117_1 = None
    permute_32: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_21, [0, 3, 1, 2]);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_33: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    amax_2: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_33, [-1], True)
    sub_7: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_33, amax_2);  clone_33 = amax_2 = None
    exp_2: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_33: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_2, [0, 2, 3, 1]);  div_2 = None
    permute_34: "f32[16, 16]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    clone_34: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
    view_49: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_34, [2654208, 16]);  clone_34 = None
    mm_5: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_49, permute_34);  view_49 = permute_34 = None
    view_50: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_5, [8, 576, 576, 16]);  mm_5 = None
    add_22: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_50, arg119_1);  view_50 = arg119_1 = None
    permute_35: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_22, [0, 3, 1, 2]);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_35: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_35);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_10: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_35, [8, 16, 576, 576]);  clone_35 = None
    clone_36: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_51: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_36, [128, 576, 576]);  clone_36 = None
    expand_11: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_8, [8, 16, 576, 48]);  select_8 = None
    clone_37: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_52: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_37, [128, 576, 48]);  clone_37 = None
    bmm_5: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_51, view_52);  view_51 = view_52 = None
    view_53: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_5, [8, 16, 576, 48]);  bmm_5 = None
    permute_36: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    clone_38: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    view_54: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_38, [8, 576, 768]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_55: "f32[4608, 768]" = torch.ops.aten.view.default(view_54, [4608, 768]);  view_54 = None
    permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_9: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg121_1, view_55, permute_37);  arg121_1 = view_55 = permute_37 = None
    view_56: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_9, [8, 576, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_39: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_23: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg5_1, clone_39);  arg5_1 = clone_39 = None
    add_23: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_18, mul_23);  add_18 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_40: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_23, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 576, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 576, 1]" = var_mean_5[1];  var_mean_5 = None
    add_24: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_8: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_40, getitem_11);  clone_40 = getitem_11 = None
    mul_24: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_25: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg122_1);  mul_24 = arg122_1 = None
    add_25: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_25, arg123_1);  mul_25 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[4608, 768]" = torch.ops.aten.view.default(add_25, [4608, 768]);  add_25 = None
    permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_10: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg125_1, view_57, permute_38);  arg125_1 = view_57 = permute_38 = None
    view_58: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 576, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_27: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_2: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_26: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_28: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_26);  mul_26 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_41: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_41, [4608, 3072]);  clone_41 = None
    permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_11: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg127_1, view_59, permute_39);  arg127_1 = view_59 = permute_39 = None
    view_60: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_11, [8, 576, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_42: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_29: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg6_1, clone_42);  arg6_1 = clone_42 = None
    add_27: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_23, mul_29);  add_23 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_43: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_43, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 576, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 576, 1]" = var_mean_6[1];  var_mean_6 = None
    add_28: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_9: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_43, getitem_13);  clone_43 = getitem_13 = None
    mul_30: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_31: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg128_1);  mul_30 = arg128_1 = None
    add_29: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_31, arg129_1);  mul_31 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_61: "f32[4608, 768]" = torch.ops.aten.view.default(add_29, [4608, 768]);  add_29 = None
    permute_40: "f32[768, 2304]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_12: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg131_1, view_61, permute_40);  arg131_1 = view_61 = permute_40 = None
    view_62: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 576, 2304]);  addmm_12 = None
    view_63: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_62, [8, 576, 3, 16, 48]);  view_62 = None
    permute_41: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_9: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_41, 0, 0)
    mul_32: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_9, 0.14433756729740643);  select_9 = None
    select_10: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_41, 0, 1)
    select_11: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_41, 0, 2);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_42: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_10, [0, 1, 3, 2]);  select_10 = None
    expand_12: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_32, [8, 16, 576, 48]);  mul_32 = None
    clone_44: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_64: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_44, [128, 576, 48]);  clone_44 = None
    expand_13: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_42, [8, 16, 48, 576]);  permute_42 = None
    clone_45: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_65: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_45, [128, 48, 576]);  clone_45 = None
    bmm_6: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
    view_66: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_6, [8, 16, 576, 576]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_43: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_66, [0, 2, 3, 1]);  view_66 = None
    permute_44: "f32[16, 16]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    clone_46: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_67: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_46, [2654208, 16]);  clone_46 = None
    mm_6: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_67, permute_44);  view_67 = permute_44 = None
    view_68: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_6, [8, 576, 576, 16]);  mm_6 = None
    add_30: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_68, arg133_1);  view_68 = arg133_1 = None
    permute_45: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_30, [0, 3, 1, 2]);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_47: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    amax_3: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_47, [-1], True)
    sub_10: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_47, amax_3);  clone_47 = amax_3 = None
    exp_3: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_46: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_3, [0, 2, 3, 1]);  div_3 = None
    permute_47: "f32[16, 16]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    clone_48: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_69: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_48, [2654208, 16]);  clone_48 = None
    mm_7: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_69, permute_47);  view_69 = permute_47 = None
    view_70: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_7, [8, 576, 576, 16]);  mm_7 = None
    add_31: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_70, arg135_1);  view_70 = arg135_1 = None
    permute_48: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_31, [0, 3, 1, 2]);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_49: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_48);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_14: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_49, [8, 16, 576, 576]);  clone_49 = None
    clone_50: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_71: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_50, [128, 576, 576]);  clone_50 = None
    expand_15: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_11, [8, 16, 576, 48]);  select_11 = None
    clone_51: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_72: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_51, [128, 576, 48]);  clone_51 = None
    bmm_7: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_71, view_72);  view_71 = view_72 = None
    view_73: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_7, [8, 16, 576, 48]);  bmm_7 = None
    permute_49: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    clone_52: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_74: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_52, [8, 576, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_75: "f32[4608, 768]" = torch.ops.aten.view.default(view_74, [4608, 768]);  view_74 = None
    permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_13: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg137_1, view_75, permute_50);  arg137_1 = view_75 = permute_50 = None
    view_76: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_13, [8, 576, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_53: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_33: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg7_1, clone_53);  arg7_1 = clone_53 = None
    add_32: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_27, mul_33);  add_27 = mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_54: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_54, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 576, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 576, 1]" = var_mean_7[1];  var_mean_7 = None
    add_33: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_11: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_54, getitem_15);  clone_54 = getitem_15 = None
    mul_34: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_35: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg138_1);  mul_34 = arg138_1 = None
    add_34: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_35, arg139_1);  mul_35 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[4608, 768]" = torch.ops.aten.view.default(add_34, [4608, 768]);  add_34 = None
    permute_51: "f32[768, 3072]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_14: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg141_1, view_77, permute_51);  arg141_1 = view_77 = permute_51 = None
    view_78: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 576, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_36: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_37: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_3: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_35: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_38: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_35);  mul_36 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_79: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_55, [4608, 3072]);  clone_55 = None
    permute_52: "f32[3072, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_15: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg143_1, view_79, permute_52);  arg143_1 = view_79 = permute_52 = None
    view_80: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_15, [8, 576, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_39: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg8_1, clone_56);  arg8_1 = clone_56 = None
    add_36: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_32, mul_39);  add_32 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_57: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 576, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 576, 1]" = var_mean_8[1];  var_mean_8 = None
    add_37: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_12: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_57, getitem_17);  clone_57 = getitem_17 = None
    mul_40: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_41: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg144_1);  mul_40 = arg144_1 = None
    add_38: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_41, arg145_1);  mul_41 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_81: "f32[4608, 768]" = torch.ops.aten.view.default(add_38, [4608, 768]);  add_38 = None
    permute_53: "f32[768, 2304]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_16: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg147_1, view_81, permute_53);  arg147_1 = view_81 = permute_53 = None
    view_82: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 576, 2304]);  addmm_16 = None
    view_83: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_82, [8, 576, 3, 16, 48]);  view_82 = None
    permute_54: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_12: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_54, 0, 0)
    mul_42: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_12, 0.14433756729740643);  select_12 = None
    select_13: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_54, 0, 1)
    select_14: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_54, 0, 2);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_55: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_13, [0, 1, 3, 2]);  select_13 = None
    expand_16: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_42, [8, 16, 576, 48]);  mul_42 = None
    clone_58: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_84: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_58, [128, 576, 48]);  clone_58 = None
    expand_17: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_55, [8, 16, 48, 576]);  permute_55 = None
    clone_59: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_85: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_59, [128, 48, 576]);  clone_59 = None
    bmm_8: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_84, view_85);  view_84 = view_85 = None
    view_86: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_8, [8, 16, 576, 576]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_56: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_86, [0, 2, 3, 1]);  view_86 = None
    permute_57: "f32[16, 16]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    clone_60: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_87: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_60, [2654208, 16]);  clone_60 = None
    mm_8: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_87, permute_57);  view_87 = permute_57 = None
    view_88: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_8, [8, 576, 576, 16]);  mm_8 = None
    add_39: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_88, arg149_1);  view_88 = arg149_1 = None
    permute_58: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_39, [0, 3, 1, 2]);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_61: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    amax_4: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_61, [-1], True)
    sub_13: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_61, amax_4);  clone_61 = amax_4 = None
    exp_4: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_59: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_4, [0, 2, 3, 1]);  div_4 = None
    permute_60: "f32[16, 16]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    clone_62: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    view_89: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_62, [2654208, 16]);  clone_62 = None
    mm_9: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_89, permute_60);  view_89 = permute_60 = None
    view_90: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_9, [8, 576, 576, 16]);  mm_9 = None
    add_40: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_90, arg151_1);  view_90 = arg151_1 = None
    permute_61: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_40, [0, 3, 1, 2]);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_63: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_61);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_18: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_63, [8, 16, 576, 576]);  clone_63 = None
    clone_64: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_91: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_64, [128, 576, 576]);  clone_64 = None
    expand_19: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_14, [8, 16, 576, 48]);  select_14 = None
    clone_65: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_92: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_65, [128, 576, 48]);  clone_65 = None
    bmm_9: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_91, view_92);  view_91 = view_92 = None
    view_93: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_9, [8, 16, 576, 48]);  bmm_9 = None
    permute_62: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    clone_66: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_94: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_66, [8, 576, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_95: "f32[4608, 768]" = torch.ops.aten.view.default(view_94, [4608, 768]);  view_94 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_17: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg153_1, view_95, permute_63);  arg153_1 = view_95 = permute_63 = None
    view_96: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_17, [8, 576, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_67: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_43: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg9_1, clone_67);  arg9_1 = clone_67 = None
    add_41: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_36, mul_43);  add_36 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_68: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_68, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 576, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 576, 1]" = var_mean_9[1];  var_mean_9 = None
    add_42: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_14: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_68, getitem_19);  clone_68 = getitem_19 = None
    mul_44: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_45: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg154_1);  mul_44 = arg154_1 = None
    add_43: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_45, arg155_1);  mul_45 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_97: "f32[4608, 768]" = torch.ops.aten.view.default(add_43, [4608, 768]);  add_43 = None
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_18: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg157_1, view_97, permute_64);  arg157_1 = view_97 = permute_64 = None
    view_98: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 576, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_47: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_4: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_44: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_48: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_44);  mul_46 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_69: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_99: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_69, [4608, 3072]);  clone_69 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_19: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg159_1, view_99, permute_65);  arg159_1 = view_99 = permute_65 = None
    view_100: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_19, [8, 576, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_70: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_49: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg10_1, clone_70);  arg10_1 = clone_70 = None
    add_45: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_41, mul_49);  add_41 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_71: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_45, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_71, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 576, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 576, 1]" = var_mean_10[1];  var_mean_10 = None
    add_46: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_15: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_71, getitem_21);  clone_71 = getitem_21 = None
    mul_50: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_51: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg160_1);  mul_50 = arg160_1 = None
    add_47: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_51, arg161_1);  mul_51 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_101: "f32[4608, 768]" = torch.ops.aten.view.default(add_47, [4608, 768]);  add_47 = None
    permute_66: "f32[768, 2304]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_20: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg163_1, view_101, permute_66);  arg163_1 = view_101 = permute_66 = None
    view_102: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 576, 2304]);  addmm_20 = None
    view_103: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_102, [8, 576, 3, 16, 48]);  view_102 = None
    permute_67: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_15: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_67, 0, 0)
    mul_52: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_15, 0.14433756729740643);  select_15 = None
    select_16: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_67, 0, 1)
    select_17: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_67, 0, 2);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_68: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_16, [0, 1, 3, 2]);  select_16 = None
    expand_20: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_52, [8, 16, 576, 48]);  mul_52 = None
    clone_72: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_104: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_72, [128, 576, 48]);  clone_72 = None
    expand_21: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_68, [8, 16, 48, 576]);  permute_68 = None
    clone_73: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_105: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_73, [128, 48, 576]);  clone_73 = None
    bmm_10: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_104, view_105);  view_104 = view_105 = None
    view_106: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_10, [8, 16, 576, 576]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_69: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_106, [0, 2, 3, 1]);  view_106 = None
    permute_70: "f32[16, 16]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    clone_74: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_107: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_74, [2654208, 16]);  clone_74 = None
    mm_10: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_107, permute_70);  view_107 = permute_70 = None
    view_108: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_10, [8, 576, 576, 16]);  mm_10 = None
    add_48: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_108, arg165_1);  view_108 = arg165_1 = None
    permute_71: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_48, [0, 3, 1, 2]);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_75: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    amax_5: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_75, [-1], True)
    sub_16: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_75, amax_5);  clone_75 = amax_5 = None
    exp_5: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_72: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_5, [0, 2, 3, 1]);  div_5 = None
    permute_73: "f32[16, 16]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    clone_76: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    view_109: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_76, [2654208, 16]);  clone_76 = None
    mm_11: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_109, permute_73);  view_109 = permute_73 = None
    view_110: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_11, [8, 576, 576, 16]);  mm_11 = None
    add_49: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_110, arg167_1);  view_110 = arg167_1 = None
    permute_74: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_49, [0, 3, 1, 2]);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_77: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_74);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_22: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_77, [8, 16, 576, 576]);  clone_77 = None
    clone_78: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_111: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_78, [128, 576, 576]);  clone_78 = None
    expand_23: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_17, [8, 16, 576, 48]);  select_17 = None
    clone_79: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_112: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_79, [128, 576, 48]);  clone_79 = None
    bmm_11: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_111, view_112);  view_111 = view_112 = None
    view_113: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_11, [8, 16, 576, 48]);  bmm_11 = None
    permute_75: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    clone_80: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_114: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_80, [8, 576, 768]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_115: "f32[4608, 768]" = torch.ops.aten.view.default(view_114, [4608, 768]);  view_114 = None
    permute_76: "f32[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_21: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg169_1, view_115, permute_76);  arg169_1 = view_115 = permute_76 = None
    view_116: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_21, [8, 576, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_81: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_53: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg11_1, clone_81);  arg11_1 = clone_81 = None
    add_50: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_45, mul_53);  add_45 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_82: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 576, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 576, 1]" = var_mean_11[1];  var_mean_11 = None
    add_51: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_17: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_82, getitem_23);  clone_82 = getitem_23 = None
    mul_54: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_55: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg170_1);  mul_54 = arg170_1 = None
    add_52: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_55, arg171_1);  mul_55 = arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[4608, 768]" = torch.ops.aten.view.default(add_52, [4608, 768]);  add_52 = None
    permute_77: "f32[768, 3072]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_22: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg173_1, view_117, permute_77);  arg173_1 = view_117 = permute_77 = None
    view_118: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 576, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_56: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_57: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_5: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_53: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_58: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_56, add_53);  mul_56 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_83: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_83, [4608, 3072]);  clone_83 = None
    permute_78: "f32[3072, 768]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_23: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg175_1, view_119, permute_78);  arg175_1 = view_119 = permute_78 = None
    view_120: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_23, [8, 576, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_84: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_59: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg12_1, clone_84);  arg12_1 = clone_84 = None
    add_54: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_50, mul_59);  add_50 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_85: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_54, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 576, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 576, 1]" = var_mean_12[1];  var_mean_12 = None
    add_55: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_18: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_85, getitem_25);  clone_85 = getitem_25 = None
    mul_60: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_61: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg176_1);  mul_60 = arg176_1 = None
    add_56: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_61, arg177_1);  mul_61 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_121: "f32[4608, 768]" = torch.ops.aten.view.default(add_56, [4608, 768]);  add_56 = None
    permute_79: "f32[768, 2304]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_24: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg179_1, view_121, permute_79);  arg179_1 = view_121 = permute_79 = None
    view_122: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 576, 2304]);  addmm_24 = None
    view_123: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_122, [8, 576, 3, 16, 48]);  view_122 = None
    permute_80: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_18: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_80, 0, 0)
    mul_62: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_18, 0.14433756729740643);  select_18 = None
    select_19: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_80, 0, 1)
    select_20: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_80, 0, 2);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_81: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_19, [0, 1, 3, 2]);  select_19 = None
    expand_24: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_62, [8, 16, 576, 48]);  mul_62 = None
    clone_86: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_124: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_86, [128, 576, 48]);  clone_86 = None
    expand_25: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_81, [8, 16, 48, 576]);  permute_81 = None
    clone_87: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_125: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_87, [128, 48, 576]);  clone_87 = None
    bmm_12: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
    view_126: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_12, [8, 16, 576, 576]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_82: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_126, [0, 2, 3, 1]);  view_126 = None
    permute_83: "f32[16, 16]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    clone_88: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_127: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_88, [2654208, 16]);  clone_88 = None
    mm_12: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_127, permute_83);  view_127 = permute_83 = None
    view_128: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_12, [8, 576, 576, 16]);  mm_12 = None
    add_57: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_128, arg181_1);  view_128 = arg181_1 = None
    permute_84: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_57, [0, 3, 1, 2]);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_89: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    amax_6: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_89, [-1], True)
    sub_19: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_89, amax_6);  clone_89 = amax_6 = None
    exp_6: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_85: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_6, [0, 2, 3, 1]);  div_6 = None
    permute_86: "f32[16, 16]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    clone_90: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_129: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_90, [2654208, 16]);  clone_90 = None
    mm_13: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_129, permute_86);  view_129 = permute_86 = None
    view_130: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_13, [8, 576, 576, 16]);  mm_13 = None
    add_58: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_130, arg183_1);  view_130 = arg183_1 = None
    permute_87: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_58, [0, 3, 1, 2]);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_91: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_87);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_26: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_91, [8, 16, 576, 576]);  clone_91 = None
    clone_92: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_131: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_92, [128, 576, 576]);  clone_92 = None
    expand_27: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_20, [8, 16, 576, 48]);  select_20 = None
    clone_93: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_132: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_93, [128, 576, 48]);  clone_93 = None
    bmm_13: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_131, view_132);  view_131 = view_132 = None
    view_133: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_13, [8, 16, 576, 48]);  bmm_13 = None
    permute_88: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    clone_94: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    view_134: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_94, [8, 576, 768]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_135: "f32[4608, 768]" = torch.ops.aten.view.default(view_134, [4608, 768]);  view_134 = None
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_25: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg185_1, view_135, permute_89);  arg185_1 = view_135 = permute_89 = None
    view_136: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_25, [8, 576, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_95: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_136);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_63: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg13_1, clone_95);  arg13_1 = clone_95 = None
    add_59: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_54, mul_63);  add_54 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_96: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 576, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 576, 1]" = var_mean_13[1];  var_mean_13 = None
    add_60: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_20: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_27);  clone_96 = getitem_27 = None
    mul_64: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_65: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg186_1);  mul_64 = arg186_1 = None
    add_61: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_65, arg187_1);  mul_65 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_137: "f32[4608, 768]" = torch.ops.aten.view.default(add_61, [4608, 768]);  add_61 = None
    permute_90: "f32[768, 3072]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_26: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg189_1, view_137, permute_90);  arg189_1 = view_137 = permute_90 = None
    view_138: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 576, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_66: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
    mul_67: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
    erf_6: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_62: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_68: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_66, add_62);  mul_66 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_97: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_139: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_97, [4608, 3072]);  clone_97 = None
    permute_91: "f32[3072, 768]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_27: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg191_1, view_139, permute_91);  arg191_1 = view_139 = permute_91 = None
    view_140: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_27, [8, 576, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_98: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_69: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg14_1, clone_98);  arg14_1 = clone_98 = None
    add_63: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_59, mul_69);  add_59 = mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_99: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_63, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_99, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 576, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 576, 1]" = var_mean_14[1];  var_mean_14 = None
    add_64: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_21: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_99, getitem_29);  clone_99 = getitem_29 = None
    mul_70: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_71: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg192_1);  mul_70 = arg192_1 = None
    add_65: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_71, arg193_1);  mul_71 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_141: "f32[4608, 768]" = torch.ops.aten.view.default(add_65, [4608, 768]);  add_65 = None
    permute_92: "f32[768, 2304]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_28: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg195_1, view_141, permute_92);  arg195_1 = view_141 = permute_92 = None
    view_142: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 576, 2304]);  addmm_28 = None
    view_143: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_142, [8, 576, 3, 16, 48]);  view_142 = None
    permute_93: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_143, [2, 0, 3, 1, 4]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_21: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_93, 0, 0)
    mul_72: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_21, 0.14433756729740643);  select_21 = None
    select_22: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_93, 0, 1)
    select_23: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_93, 0, 2);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_94: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_22, [0, 1, 3, 2]);  select_22 = None
    expand_28: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_72, [8, 16, 576, 48]);  mul_72 = None
    clone_100: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_144: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_100, [128, 576, 48]);  clone_100 = None
    expand_29: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_94, [8, 16, 48, 576]);  permute_94 = None
    clone_101: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_145: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_101, [128, 48, 576]);  clone_101 = None
    bmm_14: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
    view_146: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_14, [8, 16, 576, 576]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_95: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_146, [0, 2, 3, 1]);  view_146 = None
    permute_96: "f32[16, 16]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    clone_102: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_147: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_102, [2654208, 16]);  clone_102 = None
    mm_14: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_147, permute_96);  view_147 = permute_96 = None
    view_148: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_14, [8, 576, 576, 16]);  mm_14 = None
    add_66: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_148, arg197_1);  view_148 = arg197_1 = None
    permute_97: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_66, [0, 3, 1, 2]);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_103: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    amax_7: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_103, [-1], True)
    sub_22: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_103, amax_7);  clone_103 = amax_7 = None
    exp_7: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_98: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_7, [0, 2, 3, 1]);  div_7 = None
    permute_99: "f32[16, 16]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    clone_104: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    view_149: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_104, [2654208, 16]);  clone_104 = None
    mm_15: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_149, permute_99);  view_149 = permute_99 = None
    view_150: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_15, [8, 576, 576, 16]);  mm_15 = None
    add_67: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_150, arg199_1);  view_150 = arg199_1 = None
    permute_100: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_67, [0, 3, 1, 2]);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_105: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_100);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_30: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_105, [8, 16, 576, 576]);  clone_105 = None
    clone_106: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_151: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_106, [128, 576, 576]);  clone_106 = None
    expand_31: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_23, [8, 16, 576, 48]);  select_23 = None
    clone_107: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_152: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_107, [128, 576, 48]);  clone_107 = None
    bmm_15: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_151, view_152);  view_151 = view_152 = None
    view_153: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_15, [8, 16, 576, 48]);  bmm_15 = None
    permute_101: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    clone_108: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    view_154: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_108, [8, 576, 768]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_155: "f32[4608, 768]" = torch.ops.aten.view.default(view_154, [4608, 768]);  view_154 = None
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    addmm_29: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg201_1, view_155, permute_102);  arg201_1 = view_155 = permute_102 = None
    view_156: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_29, [8, 576, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_109: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_73: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg15_1, clone_109);  arg15_1 = clone_109 = None
    add_68: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_63, mul_73);  add_63 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_110: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_110, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 576, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 576, 1]" = var_mean_15[1];  var_mean_15 = None
    add_69: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_23: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_110, getitem_31);  clone_110 = getitem_31 = None
    mul_74: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_75: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg202_1);  mul_74 = arg202_1 = None
    add_70: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_75, arg203_1);  mul_75 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_157: "f32[4608, 768]" = torch.ops.aten.view.default(add_70, [4608, 768]);  add_70 = None
    permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_30: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg205_1, view_157, permute_103);  arg205_1 = view_157 = permute_103 = None
    view_158: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 576, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_76: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_77: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_7: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_71: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_78: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_71);  mul_76 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_111: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_78);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_159: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_111, [4608, 3072]);  clone_111 = None
    permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_31: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg207_1, view_159, permute_104);  arg207_1 = view_159 = permute_104 = None
    view_160: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_31, [8, 576, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_112: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_160);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_79: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg16_1, clone_112);  arg16_1 = clone_112 = None
    add_72: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_68, mul_79);  add_68 = mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_113: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_72, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_113, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 576, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 576, 1]" = var_mean_16[1];  var_mean_16 = None
    add_73: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_24: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_113, getitem_33);  clone_113 = getitem_33 = None
    mul_80: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_81: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg208_1);  mul_80 = arg208_1 = None
    add_74: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_81, arg209_1);  mul_81 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_161: "f32[4608, 768]" = torch.ops.aten.view.default(add_74, [4608, 768]);  add_74 = None
    permute_105: "f32[768, 2304]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_32: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg211_1, view_161, permute_105);  arg211_1 = view_161 = permute_105 = None
    view_162: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 576, 2304]);  addmm_32 = None
    view_163: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_162, [8, 576, 3, 16, 48]);  view_162 = None
    permute_106: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_24: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_106, 0, 0)
    mul_82: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_24, 0.14433756729740643);  select_24 = None
    select_25: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_106, 0, 1)
    select_26: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_106, 0, 2);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_107: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_25, [0, 1, 3, 2]);  select_25 = None
    expand_32: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_82, [8, 16, 576, 48]);  mul_82 = None
    clone_114: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_164: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_114, [128, 576, 48]);  clone_114 = None
    expand_33: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_107, [8, 16, 48, 576]);  permute_107 = None
    clone_115: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_165: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_115, [128, 48, 576]);  clone_115 = None
    bmm_16: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_164, view_165);  view_164 = view_165 = None
    view_166: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_16, [8, 16, 576, 576]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_108: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_166, [0, 2, 3, 1]);  view_166 = None
    permute_109: "f32[16, 16]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    clone_116: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    view_167: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_116, [2654208, 16]);  clone_116 = None
    mm_16: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_167, permute_109);  view_167 = permute_109 = None
    view_168: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_16, [8, 576, 576, 16]);  mm_16 = None
    add_75: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_168, arg213_1);  view_168 = arg213_1 = None
    permute_110: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_75, [0, 3, 1, 2]);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_117: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    amax_8: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_117, [-1], True)
    sub_25: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_117, amax_8);  clone_117 = amax_8 = None
    exp_8: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_111: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_8, [0, 2, 3, 1]);  div_8 = None
    permute_112: "f32[16, 16]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    clone_118: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_169: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_118, [2654208, 16]);  clone_118 = None
    mm_17: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_169, permute_112);  view_169 = permute_112 = None
    view_170: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_17, [8, 576, 576, 16]);  mm_17 = None
    add_76: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_170, arg215_1);  view_170 = arg215_1 = None
    permute_113: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_76, [0, 3, 1, 2]);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_119: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_113);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_119, [8, 16, 576, 576]);  clone_119 = None
    clone_120: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_171: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_120, [128, 576, 576]);  clone_120 = None
    expand_35: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_26, [8, 16, 576, 48]);  select_26 = None
    clone_121: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_172: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_121, [128, 576, 48]);  clone_121 = None
    bmm_17: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
    view_173: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_17, [8, 16, 576, 48]);  bmm_17 = None
    permute_114: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    clone_122: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_174: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_122, [8, 576, 768]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_175: "f32[4608, 768]" = torch.ops.aten.view.default(view_174, [4608, 768]);  view_174 = None
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    addmm_33: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg217_1, view_175, permute_115);  arg217_1 = view_175 = permute_115 = None
    view_176: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_33, [8, 576, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_123: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_176);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_83: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg17_1, clone_123);  arg17_1 = clone_123 = None
    add_77: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_72, mul_83);  add_72 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_124: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_124, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 576, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 576, 1]" = var_mean_17[1];  var_mean_17 = None
    add_78: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_26: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_124, getitem_35);  clone_124 = getitem_35 = None
    mul_84: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_85: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg218_1);  mul_84 = arg218_1 = None
    add_79: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_85, arg219_1);  mul_85 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_177: "f32[4608, 768]" = torch.ops.aten.view.default(add_79, [4608, 768]);  add_79 = None
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    addmm_34: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg221_1, view_177, permute_116);  arg221_1 = view_177 = permute_116 = None
    view_178: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 576, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    mul_87: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
    erf_8: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_80: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_88: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_80);  mul_86 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_125: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_179: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_125, [4608, 3072]);  clone_125 = None
    permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
    addmm_35: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg223_1, view_179, permute_117);  arg223_1 = view_179 = permute_117 = None
    view_180: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_35, [8, 576, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_126: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_89: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg18_1, clone_126);  arg18_1 = clone_126 = None
    add_81: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_77, mul_89);  add_77 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_127: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_81, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 576, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 576, 1]" = var_mean_18[1];  var_mean_18 = None
    add_82: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_27: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_127, getitem_37);  clone_127 = getitem_37 = None
    mul_90: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_91: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg224_1);  mul_90 = arg224_1 = None
    add_83: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_91, arg225_1);  mul_91 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_181: "f32[4608, 768]" = torch.ops.aten.view.default(add_83, [4608, 768]);  add_83 = None
    permute_118: "f32[768, 2304]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    addmm_36: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg227_1, view_181, permute_118);  arg227_1 = view_181 = permute_118 = None
    view_182: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 576, 2304]);  addmm_36 = None
    view_183: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_182, [8, 576, 3, 16, 48]);  view_182 = None
    permute_119: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_183, [2, 0, 3, 1, 4]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_27: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_119, 0, 0)
    mul_92: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_27, 0.14433756729740643);  select_27 = None
    select_28: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_119, 0, 1)
    select_29: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_119, 0, 2);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_120: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_28, [0, 1, 3, 2]);  select_28 = None
    expand_36: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_92, [8, 16, 576, 48]);  mul_92 = None
    clone_128: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_184: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_128, [128, 576, 48]);  clone_128 = None
    expand_37: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_120, [8, 16, 48, 576]);  permute_120 = None
    clone_129: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_185: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_129, [128, 48, 576]);  clone_129 = None
    bmm_18: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_184, view_185);  view_184 = view_185 = None
    view_186: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_18, [8, 16, 576, 576]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_121: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_186, [0, 2, 3, 1]);  view_186 = None
    permute_122: "f32[16, 16]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    clone_130: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_187: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_130, [2654208, 16]);  clone_130 = None
    mm_18: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_187, permute_122);  view_187 = permute_122 = None
    view_188: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_18, [8, 576, 576, 16]);  mm_18 = None
    add_84: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_188, arg229_1);  view_188 = arg229_1 = None
    permute_123: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_84, [0, 3, 1, 2]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_131: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    amax_9: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_131, [-1], True)
    sub_28: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_131, amax_9);  clone_131 = amax_9 = None
    exp_9: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_124: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_9, [0, 2, 3, 1]);  div_9 = None
    permute_125: "f32[16, 16]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    clone_132: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_189: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_132, [2654208, 16]);  clone_132 = None
    mm_19: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_189, permute_125);  view_189 = permute_125 = None
    view_190: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_19, [8, 576, 576, 16]);  mm_19 = None
    add_85: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_190, arg231_1);  view_190 = arg231_1 = None
    permute_126: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_85, [0, 3, 1, 2]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_133: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_126);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_38: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_133, [8, 16, 576, 576]);  clone_133 = None
    clone_134: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_191: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_134, [128, 576, 576]);  clone_134 = None
    expand_39: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_29, [8, 16, 576, 48]);  select_29 = None
    clone_135: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_192: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_135, [128, 576, 48]);  clone_135 = None
    bmm_19: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_191, view_192);  view_191 = view_192 = None
    view_193: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_19, [8, 16, 576, 48]);  bmm_19 = None
    permute_127: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    clone_136: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_194: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_136, [8, 576, 768]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_195: "f32[4608, 768]" = torch.ops.aten.view.default(view_194, [4608, 768]);  view_194 = None
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
    addmm_37: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg233_1, view_195, permute_128);  arg233_1 = view_195 = permute_128 = None
    view_196: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_37, [8, 576, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_137: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_196);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_93: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg19_1, clone_137);  arg19_1 = clone_137 = None
    add_86: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_81, mul_93);  add_81 = mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_138: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_138, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 576, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 576, 1]" = var_mean_19[1];  var_mean_19 = None
    add_87: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_29: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_138, getitem_39);  clone_138 = getitem_39 = None
    mul_94: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_95: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg234_1);  mul_94 = arg234_1 = None
    add_88: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_95, arg235_1);  mul_95 = arg235_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_197: "f32[4608, 768]" = torch.ops.aten.view.default(add_88, [4608, 768]);  add_88 = None
    permute_129: "f32[768, 3072]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    addmm_38: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg237_1, view_197, permute_129);  arg237_1 = view_197 = permute_129 = None
    view_198: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 576, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_96: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_198, 0.5)
    mul_97: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476);  view_198 = None
    erf_9: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_89: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_98: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_96, add_89);  mul_96 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_139: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_199: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_139, [4608, 3072]);  clone_139 = None
    permute_130: "f32[3072, 768]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    addmm_39: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg239_1, view_199, permute_130);  arg239_1 = view_199 = permute_130 = None
    view_200: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_39, [8, 576, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_140: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_200);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_99: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg20_1, clone_140);  arg20_1 = clone_140 = None
    add_90: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_86, mul_99);  add_86 = mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_141: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_141, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 576, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 576, 1]" = var_mean_20[1];  var_mean_20 = None
    add_91: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_30: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_141, getitem_41);  clone_141 = getitem_41 = None
    mul_100: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_101: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_100, arg240_1);  mul_100 = arg240_1 = None
    add_92: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_101, arg241_1);  mul_101 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_201: "f32[4608, 768]" = torch.ops.aten.view.default(add_92, [4608, 768]);  add_92 = None
    permute_131: "f32[768, 2304]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
    addmm_40: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg243_1, view_201, permute_131);  arg243_1 = view_201 = permute_131 = None
    view_202: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 576, 2304]);  addmm_40 = None
    view_203: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_202, [8, 576, 3, 16, 48]);  view_202 = None
    permute_132: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_203, [2, 0, 3, 1, 4]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_30: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_132, 0, 0)
    mul_102: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_30, 0.14433756729740643);  select_30 = None
    select_31: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_132, 0, 1)
    select_32: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_132, 0, 2);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_133: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_31, [0, 1, 3, 2]);  select_31 = None
    expand_40: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_102, [8, 16, 576, 48]);  mul_102 = None
    clone_142: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_204: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_142, [128, 576, 48]);  clone_142 = None
    expand_41: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_133, [8, 16, 48, 576]);  permute_133 = None
    clone_143: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_205: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_143, [128, 48, 576]);  clone_143 = None
    bmm_20: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
    view_206: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_20, [8, 16, 576, 576]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_134: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_206, [0, 2, 3, 1]);  view_206 = None
    permute_135: "f32[16, 16]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    clone_144: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    view_207: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_144, [2654208, 16]);  clone_144 = None
    mm_20: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_207, permute_135);  view_207 = permute_135 = None
    view_208: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_20, [8, 576, 576, 16]);  mm_20 = None
    add_93: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_208, arg245_1);  view_208 = arg245_1 = None
    permute_136: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_93, [0, 3, 1, 2]);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_145: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    amax_10: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_145, [-1], True)
    sub_31: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_145, amax_10);  clone_145 = amax_10 = None
    exp_10: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_137: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_10, [0, 2, 3, 1]);  div_10 = None
    permute_138: "f32[16, 16]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    clone_146: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_209: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_146, [2654208, 16]);  clone_146 = None
    mm_21: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_209, permute_138);  view_209 = permute_138 = None
    view_210: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_21, [8, 576, 576, 16]);  mm_21 = None
    add_94: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_210, arg247_1);  view_210 = arg247_1 = None
    permute_139: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_94, [0, 3, 1, 2]);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_147: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_139);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_42: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_147, [8, 16, 576, 576]);  clone_147 = None
    clone_148: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_211: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_148, [128, 576, 576]);  clone_148 = None
    expand_43: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_32, [8, 16, 576, 48]);  select_32 = None
    clone_149: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_212: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_149, [128, 576, 48]);  clone_149 = None
    bmm_21: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_211, view_212);  view_211 = view_212 = None
    view_213: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_21, [8, 16, 576, 48]);  bmm_21 = None
    permute_140: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    clone_150: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_214: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_150, [8, 576, 768]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_215: "f32[4608, 768]" = torch.ops.aten.view.default(view_214, [4608, 768]);  view_214 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    addmm_41: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg249_1, view_215, permute_141);  arg249_1 = view_215 = permute_141 = None
    view_216: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_41, [8, 576, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_151: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_216);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_103: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg21_1, clone_151);  arg21_1 = clone_151 = None
    add_95: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_90, mul_103);  add_90 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_152: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_95, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_152, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 576, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 576, 1]" = var_mean_21[1];  var_mean_21 = None
    add_96: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_21: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_32: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_152, getitem_43);  clone_152 = getitem_43 = None
    mul_104: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_105: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_104, arg250_1);  mul_104 = arg250_1 = None
    add_97: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_105, arg251_1);  mul_105 = arg251_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_217: "f32[4608, 768]" = torch.ops.aten.view.default(add_97, [4608, 768]);  add_97 = None
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
    addmm_42: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg253_1, view_217, permute_142);  arg253_1 = view_217 = permute_142 = None
    view_218: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 576, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_218, 0.5)
    mul_107: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_218, 0.7071067811865476);  view_218 = None
    erf_10: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_98: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_108: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_106, add_98);  mul_106 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_153: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_153, [4608, 3072]);  clone_153 = None
    permute_143: "f32[3072, 768]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_43: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg255_1, view_219, permute_143);  arg255_1 = view_219 = permute_143 = None
    view_220: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_43, [8, 576, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_154: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_220);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_109: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg22_1, clone_154);  arg22_1 = clone_154 = None
    add_99: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_95, mul_109);  add_95 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_155: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_155, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 576, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 576, 1]" = var_mean_22[1];  var_mean_22 = None
    add_100: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_22: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_33: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_155, getitem_45);  clone_155 = getitem_45 = None
    mul_110: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_111: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_110, arg256_1);  mul_110 = arg256_1 = None
    add_101: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_111, arg257_1);  mul_111 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_221: "f32[4608, 768]" = torch.ops.aten.view.default(add_101, [4608, 768]);  add_101 = None
    permute_144: "f32[768, 2304]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
    addmm_44: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg259_1, view_221, permute_144);  arg259_1 = view_221 = permute_144 = None
    view_222: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 576, 2304]);  addmm_44 = None
    view_223: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_222, [8, 576, 3, 16, 48]);  view_222 = None
    permute_145: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_223, [2, 0, 3, 1, 4]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_33: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_145, 0, 0)
    mul_112: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_33, 0.14433756729740643);  select_33 = None
    select_34: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_145, 0, 1)
    select_35: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_145, 0, 2);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_146: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_34, [0, 1, 3, 2]);  select_34 = None
    expand_44: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_112, [8, 16, 576, 48]);  mul_112 = None
    clone_156: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_224: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_156, [128, 576, 48]);  clone_156 = None
    expand_45: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_146, [8, 16, 48, 576]);  permute_146 = None
    clone_157: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_225: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_157, [128, 48, 576]);  clone_157 = None
    bmm_22: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
    view_226: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_22, [8, 16, 576, 576]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_147: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_226, [0, 2, 3, 1]);  view_226 = None
    permute_148: "f32[16, 16]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    clone_158: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_227: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_158, [2654208, 16]);  clone_158 = None
    mm_22: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_227, permute_148);  view_227 = permute_148 = None
    view_228: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_22, [8, 576, 576, 16]);  mm_22 = None
    add_102: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_228, arg261_1);  view_228 = arg261_1 = None
    permute_149: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_102, [0, 3, 1, 2]);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_159: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
    amax_11: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_159, [-1], True)
    sub_34: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_159, amax_11);  clone_159 = amax_11 = None
    exp_11: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_150: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_11, [0, 2, 3, 1]);  div_11 = None
    permute_151: "f32[16, 16]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
    clone_160: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_229: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_160, [2654208, 16]);  clone_160 = None
    mm_23: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_229, permute_151);  view_229 = permute_151 = None
    view_230: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_23, [8, 576, 576, 16]);  mm_23 = None
    add_103: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_230, arg263_1);  view_230 = arg263_1 = None
    permute_152: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_103, [0, 3, 1, 2]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_161: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_152);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_46: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_161, [8, 16, 576, 576]);  clone_161 = None
    clone_162: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_231: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_162, [128, 576, 576]);  clone_162 = None
    expand_47: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_35, [8, 16, 576, 48]);  select_35 = None
    clone_163: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_232: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_163, [128, 576, 48]);  clone_163 = None
    bmm_23: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
    view_233: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_23, [8, 16, 576, 48]);  bmm_23 = None
    permute_153: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
    clone_164: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_234: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_164, [8, 576, 768]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_235: "f32[4608, 768]" = torch.ops.aten.view.default(view_234, [4608, 768]);  view_234 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    addmm_45: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg265_1, view_235, permute_154);  arg265_1 = view_235 = permute_154 = None
    view_236: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_45, [8, 576, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_165: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_236);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_113: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg23_1, clone_165);  arg23_1 = clone_165 = None
    add_104: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_99, mul_113);  add_99 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_166: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_166, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 576, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 576, 1]" = var_mean_23[1];  var_mean_23 = None
    add_105: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_23: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_35: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_166, getitem_47);  clone_166 = getitem_47 = None
    mul_114: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_115: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_114, arg266_1);  mul_114 = arg266_1 = None
    add_106: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_115, arg267_1);  mul_115 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_237: "f32[4608, 768]" = torch.ops.aten.view.default(add_106, [4608, 768]);  add_106 = None
    permute_155: "f32[768, 3072]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
    addmm_46: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg269_1, view_237, permute_155);  arg269_1 = view_237 = permute_155 = None
    view_238: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 576, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_116: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_238, 0.5)
    mul_117: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_238, 0.7071067811865476);  view_238 = None
    erf_11: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_107: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_118: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_116, add_107);  mul_116 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_167: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_118);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_239: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_167, [4608, 3072]);  clone_167 = None
    permute_156: "f32[3072, 768]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_47: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg271_1, view_239, permute_156);  arg271_1 = view_239 = permute_156 = None
    view_240: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_47, [8, 576, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_168: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_240);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_119: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg24_1, clone_168);  arg24_1 = clone_168 = None
    add_108: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_104, mul_119);  add_104 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_169: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_169, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 576, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 576, 1]" = var_mean_24[1];  var_mean_24 = None
    add_109: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_24: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_36: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_169, getitem_49);  clone_169 = getitem_49 = None
    mul_120: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_121: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_120, arg272_1);  mul_120 = arg272_1 = None
    add_110: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_121, arg273_1);  mul_121 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_241: "f32[4608, 768]" = torch.ops.aten.view.default(add_110, [4608, 768]);  add_110 = None
    permute_157: "f32[768, 2304]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
    addmm_48: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg275_1, view_241, permute_157);  arg275_1 = view_241 = permute_157 = None
    view_242: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_48, [8, 576, 2304]);  addmm_48 = None
    view_243: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_242, [8, 576, 3, 16, 48]);  view_242 = None
    permute_158: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_243, [2, 0, 3, 1, 4]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_36: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_158, 0, 0)
    mul_122: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_36, 0.14433756729740643);  select_36 = None
    select_37: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_158, 0, 1)
    select_38: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_158, 0, 2);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_159: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_37, [0, 1, 3, 2]);  select_37 = None
    expand_48: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_122, [8, 16, 576, 48]);  mul_122 = None
    clone_170: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_244: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_170, [128, 576, 48]);  clone_170 = None
    expand_49: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_159, [8, 16, 48, 576]);  permute_159 = None
    clone_171: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_245: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_171, [128, 48, 576]);  clone_171 = None
    bmm_24: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
    view_246: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_24, [8, 16, 576, 576]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_160: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_246, [0, 2, 3, 1]);  view_246 = None
    permute_161: "f32[16, 16]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    clone_172: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_247: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_172, [2654208, 16]);  clone_172 = None
    mm_24: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_247, permute_161);  view_247 = permute_161 = None
    view_248: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_24, [8, 576, 576, 16]);  mm_24 = None
    add_111: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_248, arg277_1);  view_248 = arg277_1 = None
    permute_162: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_111, [0, 3, 1, 2]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_173: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    amax_12: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_173, [-1], True)
    sub_37: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_173, amax_12);  clone_173 = amax_12 = None
    exp_12: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_13: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_163: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_12, [0, 2, 3, 1]);  div_12 = None
    permute_164: "f32[16, 16]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    clone_174: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_249: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_174, [2654208, 16]);  clone_174 = None
    mm_25: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_249, permute_164);  view_249 = permute_164 = None
    view_250: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_25, [8, 576, 576, 16]);  mm_25 = None
    add_112: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_250, arg279_1);  view_250 = arg279_1 = None
    permute_165: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_112, [0, 3, 1, 2]);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_175: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_165);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_50: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_175, [8, 16, 576, 576]);  clone_175 = None
    clone_176: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_251: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_176, [128, 576, 576]);  clone_176 = None
    expand_51: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_38, [8, 16, 576, 48]);  select_38 = None
    clone_177: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_252: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_177, [128, 576, 48]);  clone_177 = None
    bmm_25: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
    view_253: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_25, [8, 16, 576, 48]);  bmm_25 = None
    permute_166: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    clone_178: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    view_254: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_178, [8, 576, 768]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_255: "f32[4608, 768]" = torch.ops.aten.view.default(view_254, [4608, 768]);  view_254 = None
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
    addmm_49: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg281_1, view_255, permute_167);  arg281_1 = view_255 = permute_167 = None
    view_256: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_49, [8, 576, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_179: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_256);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_123: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg25_1, clone_179);  arg25_1 = clone_179 = None
    add_113: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_108, mul_123);  add_108 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_180: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_113, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_180, [2], correction = 0, keepdim = True)
    getitem_50: "f32[8, 576, 1]" = var_mean_25[0]
    getitem_51: "f32[8, 576, 1]" = var_mean_25[1];  var_mean_25 = None
    add_114: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_25: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_38: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_180, getitem_51);  clone_180 = getitem_51 = None
    mul_124: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_125: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_124, arg282_1);  mul_124 = arg282_1 = None
    add_115: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_125, arg283_1);  mul_125 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[4608, 768]" = torch.ops.aten.view.default(add_115, [4608, 768]);  add_115 = None
    permute_168: "f32[768, 3072]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
    addmm_50: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg285_1, view_257, permute_168);  arg285_1 = view_257 = permute_168 = None
    view_258: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_50, [8, 576, 3072]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_126: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.5)
    mul_127: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476);  view_258 = None
    erf_12: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
    add_116: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_128: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_126, add_116);  mul_126 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_181: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_128);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_259: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_181, [4608, 3072]);  clone_181 = None
    permute_169: "f32[3072, 768]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
    addmm_51: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg287_1, view_259, permute_169);  arg287_1 = view_259 = permute_169 = None
    view_260: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_51, [8, 576, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_182: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_260);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_129: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg26_1, clone_182);  arg26_1 = clone_182 = None
    add_117: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_113, mul_129);  add_113 = mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_183: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_117, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_183, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 576, 1]" = var_mean_26[0]
    getitem_53: "f32[8, 576, 1]" = var_mean_26[1];  var_mean_26 = None
    add_118: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_26: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_39: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_183, getitem_53);  clone_183 = getitem_53 = None
    mul_130: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
    mul_131: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_130, arg288_1);  mul_130 = arg288_1 = None
    add_119: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_131, arg289_1);  mul_131 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_261: "f32[4608, 768]" = torch.ops.aten.view.default(add_119, [4608, 768]);  add_119 = None
    permute_170: "f32[768, 2304]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
    addmm_52: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg291_1, view_261, permute_170);  arg291_1 = view_261 = permute_170 = None
    view_262: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_52, [8, 576, 2304]);  addmm_52 = None
    view_263: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_262, [8, 576, 3, 16, 48]);  view_262 = None
    permute_171: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_39: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_171, 0, 0)
    mul_132: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_39, 0.14433756729740643);  select_39 = None
    select_40: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_171, 0, 1)
    select_41: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_171, 0, 2);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_172: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_40, [0, 1, 3, 2]);  select_40 = None
    expand_52: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_132, [8, 16, 576, 48]);  mul_132 = None
    clone_184: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_264: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_184, [128, 576, 48]);  clone_184 = None
    expand_53: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_172, [8, 16, 48, 576]);  permute_172 = None
    clone_185: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_265: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_185, [128, 48, 576]);  clone_185 = None
    bmm_26: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_264, view_265);  view_264 = view_265 = None
    view_266: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_26, [8, 16, 576, 576]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_173: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_266, [0, 2, 3, 1]);  view_266 = None
    permute_174: "f32[16, 16]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
    clone_186: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    view_267: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_186, [2654208, 16]);  clone_186 = None
    mm_26: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_267, permute_174);  view_267 = permute_174 = None
    view_268: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_26, [8, 576, 576, 16]);  mm_26 = None
    add_120: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_268, arg293_1);  view_268 = arg293_1 = None
    permute_175: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_120, [0, 3, 1, 2]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_187: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    amax_13: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_187, [-1], True)
    sub_40: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_187, amax_13);  clone_187 = amax_13 = None
    exp_13: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_176: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_13, [0, 2, 3, 1]);  div_13 = None
    permute_177: "f32[16, 16]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
    clone_188: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    view_269: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_188, [2654208, 16]);  clone_188 = None
    mm_27: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_269, permute_177);  view_269 = permute_177 = None
    view_270: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_27, [8, 576, 576, 16]);  mm_27 = None
    add_121: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_270, arg295_1);  view_270 = arg295_1 = None
    permute_178: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_121, [0, 3, 1, 2]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_189: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_178);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_54: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_189, [8, 16, 576, 576]);  clone_189 = None
    clone_190: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_271: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_190, [128, 576, 576]);  clone_190 = None
    expand_55: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_41, [8, 16, 576, 48]);  select_41 = None
    clone_191: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_272: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_191, [128, 576, 48]);  clone_191 = None
    bmm_27: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_271, view_272);  view_271 = view_272 = None
    view_273: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_27, [8, 16, 576, 48]);  bmm_27 = None
    permute_179: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    clone_192: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_274: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_192, [8, 576, 768]);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_275: "f32[4608, 768]" = torch.ops.aten.view.default(view_274, [4608, 768]);  view_274 = None
    permute_180: "f32[768, 768]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
    addmm_53: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg297_1, view_275, permute_180);  arg297_1 = view_275 = permute_180 = None
    view_276: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_53, [8, 576, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_193: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_276);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_133: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg27_1, clone_193);  arg27_1 = clone_193 = None
    add_122: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_117, mul_133);  add_117 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_194: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_194, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 576, 1]" = var_mean_27[0]
    getitem_55: "f32[8, 576, 1]" = var_mean_27[1];  var_mean_27 = None
    add_123: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_27: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_41: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_194, getitem_55);  clone_194 = getitem_55 = None
    mul_134: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
    mul_135: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_134, arg298_1);  mul_134 = arg298_1 = None
    add_124: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_135, arg299_1);  mul_135 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_277: "f32[4608, 768]" = torch.ops.aten.view.default(add_124, [4608, 768]);  add_124 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
    addmm_54: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg301_1, view_277, permute_181);  arg301_1 = view_277 = permute_181 = None
    view_278: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_54, [8, 576, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_136: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_278, 0.5)
    mul_137: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_278, 0.7071067811865476);  view_278 = None
    erf_13: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_137);  mul_137 = None
    add_125: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_138: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_136, add_125);  mul_136 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_195: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_138);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_279: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_195, [4608, 3072]);  clone_195 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
    addmm_55: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg303_1, view_279, permute_182);  arg303_1 = view_279 = permute_182 = None
    view_280: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_55, [8, 576, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_196: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_280);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_139: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg28_1, clone_196);  arg28_1 = clone_196 = None
    add_126: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_122, mul_139);  add_122 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_197: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_126, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_197, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 576, 1]" = var_mean_28[0]
    getitem_57: "f32[8, 576, 1]" = var_mean_28[1];  var_mean_28 = None
    add_127: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_28: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_42: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_197, getitem_57);  clone_197 = getitem_57 = None
    mul_140: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
    mul_141: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_140, arg304_1);  mul_140 = arg304_1 = None
    add_128: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_141, arg305_1);  mul_141 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_281: "f32[4608, 768]" = torch.ops.aten.view.default(add_128, [4608, 768]);  add_128 = None
    permute_183: "f32[768, 2304]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
    addmm_56: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg307_1, view_281, permute_183);  arg307_1 = view_281 = permute_183 = None
    view_282: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_56, [8, 576, 2304]);  addmm_56 = None
    view_283: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_282, [8, 576, 3, 16, 48]);  view_282 = None
    permute_184: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_42: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_184, 0, 0)
    mul_142: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_42, 0.14433756729740643);  select_42 = None
    select_43: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_184, 0, 1)
    select_44: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_184, 0, 2);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_185: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_43, [0, 1, 3, 2]);  select_43 = None
    expand_56: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_142, [8, 16, 576, 48]);  mul_142 = None
    clone_198: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_284: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_198, [128, 576, 48]);  clone_198 = None
    expand_57: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_185, [8, 16, 48, 576]);  permute_185 = None
    clone_199: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_285: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_199, [128, 48, 576]);  clone_199 = None
    bmm_28: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_284, view_285);  view_284 = view_285 = None
    view_286: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_28, [8, 16, 576, 576]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_186: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_286, [0, 2, 3, 1]);  view_286 = None
    permute_187: "f32[16, 16]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
    clone_200: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_287: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_200, [2654208, 16]);  clone_200 = None
    mm_28: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_287, permute_187);  view_287 = permute_187 = None
    view_288: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_28, [8, 576, 576, 16]);  mm_28 = None
    add_129: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_288, arg309_1);  view_288 = arg309_1 = None
    permute_188: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_129, [0, 3, 1, 2]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_201: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    amax_14: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_201, [-1], True)
    sub_43: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_201, amax_14);  clone_201 = amax_14 = None
    exp_14: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_189: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_14, [0, 2, 3, 1]);  div_14 = None
    permute_190: "f32[16, 16]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
    clone_202: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_289: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_202, [2654208, 16]);  clone_202 = None
    mm_29: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_289, permute_190);  view_289 = permute_190 = None
    view_290: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_29, [8, 576, 576, 16]);  mm_29 = None
    add_130: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_290, arg311_1);  view_290 = arg311_1 = None
    permute_191: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_130, [0, 3, 1, 2]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_203: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_191);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_58: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_203, [8, 16, 576, 576]);  clone_203 = None
    clone_204: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_291: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_204, [128, 576, 576]);  clone_204 = None
    expand_59: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_44, [8, 16, 576, 48]);  select_44 = None
    clone_205: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_292: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_205, [128, 576, 48]);  clone_205 = None
    bmm_29: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_291, view_292);  view_291 = view_292 = None
    view_293: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_29, [8, 16, 576, 48]);  bmm_29 = None
    permute_192: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    clone_206: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
    view_294: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_206, [8, 576, 768]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_295: "f32[4608, 768]" = torch.ops.aten.view.default(view_294, [4608, 768]);  view_294 = None
    permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
    addmm_57: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg313_1, view_295, permute_193);  arg313_1 = view_295 = permute_193 = None
    view_296: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_57, [8, 576, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_207: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_296);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_143: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg29_1, clone_207);  arg29_1 = clone_207 = None
    add_131: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_126, mul_143);  add_126 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_208: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_131, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_208, [2], correction = 0, keepdim = True)
    getitem_58: "f32[8, 576, 1]" = var_mean_29[0]
    getitem_59: "f32[8, 576, 1]" = var_mean_29[1];  var_mean_29 = None
    add_132: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
    rsqrt_29: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_44: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_208, getitem_59);  clone_208 = getitem_59 = None
    mul_144: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_145: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_144, arg314_1);  mul_144 = arg314_1 = None
    add_133: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_145, arg315_1);  mul_145 = arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_297: "f32[4608, 768]" = torch.ops.aten.view.default(add_133, [4608, 768]);  add_133 = None
    permute_194: "f32[768, 3072]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
    addmm_58: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg317_1, view_297, permute_194);  arg317_1 = view_297 = permute_194 = None
    view_298: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_58, [8, 576, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_146: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_298, 0.5)
    mul_147: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_298, 0.7071067811865476);  view_298 = None
    erf_14: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_134: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_148: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_146, add_134);  mul_146 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_209: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_299: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_209, [4608, 3072]);  clone_209 = None
    permute_195: "f32[3072, 768]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
    addmm_59: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg319_1, view_299, permute_195);  arg319_1 = view_299 = permute_195 = None
    view_300: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_59, [8, 576, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_210: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_300);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_149: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg30_1, clone_210);  arg30_1 = clone_210 = None
    add_135: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_131, mul_149);  add_131 = mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_211: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_135, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_211, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 576, 1]" = var_mean_30[0]
    getitem_61: "f32[8, 576, 1]" = var_mean_30[1];  var_mean_30 = None
    add_136: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_30: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_45: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_211, getitem_61);  clone_211 = getitem_61 = None
    mul_150: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
    mul_151: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_150, arg320_1);  mul_150 = arg320_1 = None
    add_137: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_151, arg321_1);  mul_151 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_301: "f32[4608, 768]" = torch.ops.aten.view.default(add_137, [4608, 768]);  add_137 = None
    permute_196: "f32[768, 2304]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
    addmm_60: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg323_1, view_301, permute_196);  arg323_1 = view_301 = permute_196 = None
    view_302: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_60, [8, 576, 2304]);  addmm_60 = None
    view_303: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_302, [8, 576, 3, 16, 48]);  view_302 = None
    permute_197: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_303, [2, 0, 3, 1, 4]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_45: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_197, 0, 0)
    mul_152: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_45, 0.14433756729740643);  select_45 = None
    select_46: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_197, 0, 1)
    select_47: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_197, 0, 2);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_198: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_46, [0, 1, 3, 2]);  select_46 = None
    expand_60: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_152, [8, 16, 576, 48]);  mul_152 = None
    clone_212: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_304: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_212, [128, 576, 48]);  clone_212 = None
    expand_61: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_198, [8, 16, 48, 576]);  permute_198 = None
    clone_213: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_305: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_213, [128, 48, 576]);  clone_213 = None
    bmm_30: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_304, view_305);  view_304 = view_305 = None
    view_306: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_30, [8, 16, 576, 576]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_199: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_306, [0, 2, 3, 1]);  view_306 = None
    permute_200: "f32[16, 16]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
    clone_214: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_307: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_214, [2654208, 16]);  clone_214 = None
    mm_30: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_307, permute_200);  view_307 = permute_200 = None
    view_308: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_30, [8, 576, 576, 16]);  mm_30 = None
    add_138: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_308, arg325_1);  view_308 = arg325_1 = None
    permute_201: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_138, [0, 3, 1, 2]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_215: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    amax_15: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_215, [-1], True)
    sub_46: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_215, amax_15);  clone_215 = amax_15 = None
    exp_15: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_202: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_15, [0, 2, 3, 1]);  div_15 = None
    permute_203: "f32[16, 16]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
    clone_216: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    view_309: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_216, [2654208, 16]);  clone_216 = None
    mm_31: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_309, permute_203);  view_309 = permute_203 = None
    view_310: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_31, [8, 576, 576, 16]);  mm_31 = None
    add_139: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_310, arg327_1);  view_310 = arg327_1 = None
    permute_204: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_139, [0, 3, 1, 2]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_217: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_204);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_62: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_217, [8, 16, 576, 576]);  clone_217 = None
    clone_218: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    view_311: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_218, [128, 576, 576]);  clone_218 = None
    expand_63: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_47, [8, 16, 576, 48]);  select_47 = None
    clone_219: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_312: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_219, [128, 576, 48]);  clone_219 = None
    bmm_31: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_311, view_312);  view_311 = view_312 = None
    view_313: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_31, [8, 16, 576, 48]);  bmm_31 = None
    permute_205: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    clone_220: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    view_314: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_220, [8, 576, 768]);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_315: "f32[4608, 768]" = torch.ops.aten.view.default(view_314, [4608, 768]);  view_314 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
    addmm_61: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg329_1, view_315, permute_206);  arg329_1 = view_315 = permute_206 = None
    view_316: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_61, [8, 576, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_221: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_316);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_153: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg31_1, clone_221);  arg31_1 = clone_221 = None
    add_140: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_135, mul_153);  add_135 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_222: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_222, [2], correction = 0, keepdim = True)
    getitem_62: "f32[8, 576, 1]" = var_mean_31[0]
    getitem_63: "f32[8, 576, 1]" = var_mean_31[1];  var_mean_31 = None
    add_141: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_31: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_47: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_222, getitem_63);  clone_222 = getitem_63 = None
    mul_154: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_155: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_154, arg330_1);  mul_154 = arg330_1 = None
    add_142: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_155, arg331_1);  mul_155 = arg331_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_317: "f32[4608, 768]" = torch.ops.aten.view.default(add_142, [4608, 768]);  add_142 = None
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
    addmm_62: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg333_1, view_317, permute_207);  arg333_1 = view_317 = permute_207 = None
    view_318: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_62, [8, 576, 3072]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_156: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_318, 0.5)
    mul_157: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_318, 0.7071067811865476);  view_318 = None
    erf_15: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_157);  mul_157 = None
    add_143: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_158: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_156, add_143);  mul_156 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_223: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_158);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_319: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_223, [4608, 3072]);  clone_223 = None
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
    addmm_63: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg335_1, view_319, permute_208);  arg335_1 = view_319 = permute_208 = None
    view_320: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_63, [8, 576, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_224: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_320);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_159: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg32_1, clone_224);  arg32_1 = clone_224 = None
    add_144: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_140, mul_159);  add_140 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_225: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_144, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_225, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 576, 1]" = var_mean_32[0]
    getitem_65: "f32[8, 576, 1]" = var_mean_32[1];  var_mean_32 = None
    add_145: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_32: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_48: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_225, getitem_65);  clone_225 = getitem_65 = None
    mul_160: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
    mul_161: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_160, arg336_1);  mul_160 = arg336_1 = None
    add_146: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_161, arg337_1);  mul_161 = arg337_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_321: "f32[4608, 768]" = torch.ops.aten.view.default(add_146, [4608, 768]);  add_146 = None
    permute_209: "f32[768, 2304]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
    addmm_64: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg339_1, view_321, permute_209);  arg339_1 = view_321 = permute_209 = None
    view_322: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_64, [8, 576, 2304]);  addmm_64 = None
    view_323: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_322, [8, 576, 3, 16, 48]);  view_322 = None
    permute_210: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_323, [2, 0, 3, 1, 4]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_48: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_210, 0, 0)
    mul_162: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_48, 0.14433756729740643);  select_48 = None
    select_49: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_210, 0, 1)
    select_50: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_210, 0, 2);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_211: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_49, [0, 1, 3, 2]);  select_49 = None
    expand_64: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_162, [8, 16, 576, 48]);  mul_162 = None
    clone_226: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_324: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_226, [128, 576, 48]);  clone_226 = None
    expand_65: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_211, [8, 16, 48, 576]);  permute_211 = None
    clone_227: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_325: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_227, [128, 48, 576]);  clone_227 = None
    bmm_32: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_324, view_325);  view_324 = view_325 = None
    view_326: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_32, [8, 16, 576, 576]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_212: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_326, [0, 2, 3, 1]);  view_326 = None
    permute_213: "f32[16, 16]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
    clone_228: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
    view_327: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_228, [2654208, 16]);  clone_228 = None
    mm_32: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_327, permute_213);  view_327 = permute_213 = None
    view_328: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_32, [8, 576, 576, 16]);  mm_32 = None
    add_147: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_328, arg341_1);  view_328 = arg341_1 = None
    permute_214: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_147, [0, 3, 1, 2]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_229: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    amax_16: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_229, [-1], True)
    sub_49: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_229, amax_16);  clone_229 = amax_16 = None
    exp_16: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_17: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_215: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_16, [0, 2, 3, 1]);  div_16 = None
    permute_216: "f32[16, 16]" = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
    clone_230: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    view_329: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_230, [2654208, 16]);  clone_230 = None
    mm_33: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_329, permute_216);  view_329 = permute_216 = None
    view_330: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_33, [8, 576, 576, 16]);  mm_33 = None
    add_148: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_330, arg343_1);  view_330 = arg343_1 = None
    permute_217: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_148, [0, 3, 1, 2]);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_231: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_217);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_66: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_231, [8, 16, 576, 576]);  clone_231 = None
    clone_232: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_331: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_232, [128, 576, 576]);  clone_232 = None
    expand_67: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_50, [8, 16, 576, 48]);  select_50 = None
    clone_233: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_332: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_233, [128, 576, 48]);  clone_233 = None
    bmm_33: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
    view_333: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_33, [8, 16, 576, 48]);  bmm_33 = None
    permute_218: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
    clone_234: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_334: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_234, [8, 576, 768]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_335: "f32[4608, 768]" = torch.ops.aten.view.default(view_334, [4608, 768]);  view_334 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
    addmm_65: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg345_1, view_335, permute_219);  arg345_1 = view_335 = permute_219 = None
    view_336: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_65, [8, 576, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_235: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_336);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_163: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg33_1, clone_235);  arg33_1 = clone_235 = None
    add_149: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_144, mul_163);  add_144 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_236: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_149, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_236, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 576, 1]" = var_mean_33[0]
    getitem_67: "f32[8, 576, 1]" = var_mean_33[1];  var_mean_33 = None
    add_150: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_33: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_50: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_236, getitem_67);  clone_236 = getitem_67 = None
    mul_164: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
    mul_165: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_164, arg346_1);  mul_164 = arg346_1 = None
    add_151: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_165, arg347_1);  mul_165 = arg347_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_337: "f32[4608, 768]" = torch.ops.aten.view.default(add_151, [4608, 768]);  add_151 = None
    permute_220: "f32[768, 3072]" = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
    addmm_66: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg349_1, view_337, permute_220);  arg349_1 = view_337 = permute_220 = None
    view_338: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_66, [8, 576, 3072]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_166: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_338, 0.5)
    mul_167: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_338, 0.7071067811865476);  view_338 = None
    erf_16: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_152: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_168: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_166, add_152);  mul_166 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_237: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_168);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_339: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_237, [4608, 3072]);  clone_237 = None
    permute_221: "f32[3072, 768]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
    addmm_67: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg351_1, view_339, permute_221);  arg351_1 = view_339 = permute_221 = None
    view_340: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_67, [8, 576, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_238: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_340);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_169: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg34_1, clone_238);  arg34_1 = clone_238 = None
    add_153: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_149, mul_169);  add_149 = mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_239: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 576, 1]" = var_mean_34[0]
    getitem_69: "f32[8, 576, 1]" = var_mean_34[1];  var_mean_34 = None
    add_154: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_34: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_51: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_239, getitem_69);  clone_239 = getitem_69 = None
    mul_170: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
    mul_171: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_170, arg352_1);  mul_170 = arg352_1 = None
    add_155: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_171, arg353_1);  mul_171 = arg353_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_341: "f32[4608, 768]" = torch.ops.aten.view.default(add_155, [4608, 768]);  add_155 = None
    permute_222: "f32[768, 2304]" = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
    addmm_68: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg355_1, view_341, permute_222);  arg355_1 = view_341 = permute_222 = None
    view_342: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_68, [8, 576, 2304]);  addmm_68 = None
    view_343: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_342, [8, 576, 3, 16, 48]);  view_342 = None
    permute_223: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_343, [2, 0, 3, 1, 4]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_51: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_223, 0, 0)
    mul_172: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_51, 0.14433756729740643);  select_51 = None
    select_52: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_223, 0, 1)
    select_53: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_223, 0, 2);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_224: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_52, [0, 1, 3, 2]);  select_52 = None
    expand_68: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_172, [8, 16, 576, 48]);  mul_172 = None
    clone_240: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_344: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_240, [128, 576, 48]);  clone_240 = None
    expand_69: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_224, [8, 16, 48, 576]);  permute_224 = None
    clone_241: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_345: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_241, [128, 48, 576]);  clone_241 = None
    bmm_34: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_344, view_345);  view_344 = view_345 = None
    view_346: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_34, [8, 16, 576, 576]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_225: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_346, [0, 2, 3, 1]);  view_346 = None
    permute_226: "f32[16, 16]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
    clone_242: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    view_347: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_242, [2654208, 16]);  clone_242 = None
    mm_34: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_347, permute_226);  view_347 = permute_226 = None
    view_348: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_34, [8, 576, 576, 16]);  mm_34 = None
    add_156: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_348, arg357_1);  view_348 = arg357_1 = None
    permute_227: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_156, [0, 3, 1, 2]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_243: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    amax_17: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_243, [-1], True)
    sub_52: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_243, amax_17);  clone_243 = amax_17 = None
    exp_17: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_18: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_228: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_17, [0, 2, 3, 1]);  div_17 = None
    permute_229: "f32[16, 16]" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
    clone_244: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_349: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_244, [2654208, 16]);  clone_244 = None
    mm_35: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_349, permute_229);  view_349 = permute_229 = None
    view_350: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_35, [8, 576, 576, 16]);  mm_35 = None
    add_157: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_350, arg359_1);  view_350 = arg359_1 = None
    permute_230: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_157, [0, 3, 1, 2]);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_245: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_230);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_70: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_245, [8, 16, 576, 576]);  clone_245 = None
    clone_246: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_351: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_246, [128, 576, 576]);  clone_246 = None
    expand_71: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_53, [8, 16, 576, 48]);  select_53 = None
    clone_247: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_352: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_247, [128, 576, 48]);  clone_247 = None
    bmm_35: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_351, view_352);  view_351 = view_352 = None
    view_353: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_35, [8, 16, 576, 48]);  bmm_35 = None
    permute_231: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    clone_248: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_354: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_248, [8, 576, 768]);  clone_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_355: "f32[4608, 768]" = torch.ops.aten.view.default(view_354, [4608, 768]);  view_354 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
    addmm_69: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg361_1, view_355, permute_232);  arg361_1 = view_355 = permute_232 = None
    view_356: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_69, [8, 576, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_249: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_356);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_173: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg35_1, clone_249);  arg35_1 = clone_249 = None
    add_158: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_153, mul_173);  add_153 = mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_250: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_158, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_250, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 576, 1]" = var_mean_35[0]
    getitem_71: "f32[8, 576, 1]" = var_mean_35[1];  var_mean_35 = None
    add_159: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_35: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_53: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_250, getitem_71);  clone_250 = getitem_71 = None
    mul_174: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
    mul_175: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_174, arg362_1);  mul_174 = arg362_1 = None
    add_160: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_175, arg363_1);  mul_175 = arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_357: "f32[4608, 768]" = torch.ops.aten.view.default(add_160, [4608, 768]);  add_160 = None
    permute_233: "f32[768, 3072]" = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
    addmm_70: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg365_1, view_357, permute_233);  arg365_1 = view_357 = permute_233 = None
    view_358: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_70, [8, 576, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_176: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_358, 0.5)
    mul_177: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_358, 0.7071067811865476);  view_358 = None
    erf_17: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_161: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_178: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_176, add_161);  mul_176 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_251: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_359: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_251, [4608, 3072]);  clone_251 = None
    permute_234: "f32[3072, 768]" = torch.ops.aten.permute.default(arg366_1, [1, 0]);  arg366_1 = None
    addmm_71: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg367_1, view_359, permute_234);  arg367_1 = view_359 = permute_234 = None
    view_360: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_71, [8, 576, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_252: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_360);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_179: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg36_1, clone_252);  arg36_1 = clone_252 = None
    add_162: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_158, mul_179);  add_158 = mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_253: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_162, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_253, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 576, 1]" = var_mean_36[0]
    getitem_73: "f32[8, 576, 1]" = var_mean_36[1];  var_mean_36 = None
    add_163: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_36: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_54: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_253, getitem_73);  clone_253 = getitem_73 = None
    mul_180: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
    mul_181: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_180, arg368_1);  mul_180 = arg368_1 = None
    add_164: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_181, arg369_1);  mul_181 = arg369_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_361: "f32[4608, 768]" = torch.ops.aten.view.default(add_164, [4608, 768]);  add_164 = None
    permute_235: "f32[768, 2304]" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
    addmm_72: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg371_1, view_361, permute_235);  arg371_1 = view_361 = permute_235 = None
    view_362: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_72, [8, 576, 2304]);  addmm_72 = None
    view_363: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_362, [8, 576, 3, 16, 48]);  view_362 = None
    permute_236: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_54: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_236, 0, 0)
    mul_182: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_54, 0.14433756729740643);  select_54 = None
    select_55: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_236, 0, 1)
    select_56: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_236, 0, 2);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_237: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_55, [0, 1, 3, 2]);  select_55 = None
    expand_72: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_182, [8, 16, 576, 48]);  mul_182 = None
    clone_254: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_364: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_254, [128, 576, 48]);  clone_254 = None
    expand_73: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_237, [8, 16, 48, 576]);  permute_237 = None
    clone_255: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_365: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_255, [128, 48, 576]);  clone_255 = None
    bmm_36: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_364, view_365);  view_364 = view_365 = None
    view_366: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_36, [8, 16, 576, 576]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_238: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_366, [0, 2, 3, 1]);  view_366 = None
    permute_239: "f32[16, 16]" = torch.ops.aten.permute.default(arg372_1, [1, 0]);  arg372_1 = None
    clone_256: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    view_367: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_256, [2654208, 16]);  clone_256 = None
    mm_36: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_367, permute_239);  view_367 = permute_239 = None
    view_368: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_36, [8, 576, 576, 16]);  mm_36 = None
    add_165: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_368, arg373_1);  view_368 = arg373_1 = None
    permute_240: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_165, [0, 3, 1, 2]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_257: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    amax_18: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_257, [-1], True)
    sub_55: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_257, amax_18);  clone_257 = amax_18 = None
    exp_18: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_19: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_241: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_18, [0, 2, 3, 1]);  div_18 = None
    permute_242: "f32[16, 16]" = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
    clone_258: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_369: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_258, [2654208, 16]);  clone_258 = None
    mm_37: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_369, permute_242);  view_369 = permute_242 = None
    view_370: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_37, [8, 576, 576, 16]);  mm_37 = None
    add_166: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_370, arg375_1);  view_370 = arg375_1 = None
    permute_243: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_166, [0, 3, 1, 2]);  add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_259: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_243);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_74: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_259, [8, 16, 576, 576]);  clone_259 = None
    clone_260: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_371: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_260, [128, 576, 576]);  clone_260 = None
    expand_75: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_56, [8, 16, 576, 48]);  select_56 = None
    clone_261: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_372: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_261, [128, 576, 48]);  clone_261 = None
    bmm_37: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_371, view_372);  view_371 = view_372 = None
    view_373: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_37, [8, 16, 576, 48]);  bmm_37 = None
    permute_244: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    clone_262: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_374: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_262, [8, 576, 768]);  clone_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_375: "f32[4608, 768]" = torch.ops.aten.view.default(view_374, [4608, 768]);  view_374 = None
    permute_245: "f32[768, 768]" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
    addmm_73: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg377_1, view_375, permute_245);  arg377_1 = view_375 = permute_245 = None
    view_376: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_73, [8, 576, 768]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_263: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_376);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_183: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg37_1, clone_263);  arg37_1 = clone_263 = None
    add_167: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_162, mul_183);  add_162 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_264: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_264, [2], correction = 0, keepdim = True)
    getitem_74: "f32[8, 576, 1]" = var_mean_37[0]
    getitem_75: "f32[8, 576, 1]" = var_mean_37[1];  var_mean_37 = None
    add_168: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
    rsqrt_37: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_56: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_264, getitem_75);  clone_264 = getitem_75 = None
    mul_184: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
    mul_185: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_184, arg378_1);  mul_184 = arg378_1 = None
    add_169: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_185, arg379_1);  mul_185 = arg379_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_377: "f32[4608, 768]" = torch.ops.aten.view.default(add_169, [4608, 768]);  add_169 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
    addmm_74: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg381_1, view_377, permute_246);  arg381_1 = view_377 = permute_246 = None
    view_378: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_74, [8, 576, 3072]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_186: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_378, 0.5)
    mul_187: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_378, 0.7071067811865476);  view_378 = None
    erf_18: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_170: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_188: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_186, add_170);  mul_186 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_265: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_188);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_379: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_265, [4608, 3072]);  clone_265 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
    addmm_75: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg383_1, view_379, permute_247);  arg383_1 = view_379 = permute_247 = None
    view_380: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_75, [8, 576, 768]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_266: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_380);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_189: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg38_1, clone_266);  arg38_1 = clone_266 = None
    add_171: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_167, mul_189);  add_167 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_267: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_267, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 576, 1]" = var_mean_38[0]
    getitem_77: "f32[8, 576, 1]" = var_mean_38[1];  var_mean_38 = None
    add_172: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_38: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_57: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_267, getitem_77);  clone_267 = getitem_77 = None
    mul_190: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
    mul_191: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_190, arg384_1);  mul_190 = arg384_1 = None
    add_173: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_191, arg385_1);  mul_191 = arg385_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_381: "f32[4608, 768]" = torch.ops.aten.view.default(add_173, [4608, 768]);  add_173 = None
    permute_248: "f32[768, 2304]" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
    addmm_76: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg387_1, view_381, permute_248);  arg387_1 = view_381 = permute_248 = None
    view_382: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_76, [8, 576, 2304]);  addmm_76 = None
    view_383: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_382, [8, 576, 3, 16, 48]);  view_382 = None
    permute_249: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_383, [2, 0, 3, 1, 4]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_57: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_249, 0, 0)
    mul_192: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_57, 0.14433756729740643);  select_57 = None
    select_58: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_249, 0, 1)
    select_59: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_249, 0, 2);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_250: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_58, [0, 1, 3, 2]);  select_58 = None
    expand_76: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_192, [8, 16, 576, 48]);  mul_192 = None
    clone_268: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_384: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_268, [128, 576, 48]);  clone_268 = None
    expand_77: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_250, [8, 16, 48, 576]);  permute_250 = None
    clone_269: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_385: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_269, [128, 48, 576]);  clone_269 = None
    bmm_38: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
    view_386: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_38, [8, 16, 576, 576]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_251: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_386, [0, 2, 3, 1]);  view_386 = None
    permute_252: "f32[16, 16]" = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
    clone_270: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
    view_387: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_270, [2654208, 16]);  clone_270 = None
    mm_38: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_387, permute_252);  view_387 = permute_252 = None
    view_388: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_38, [8, 576, 576, 16]);  mm_38 = None
    add_174: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_388, arg389_1);  view_388 = arg389_1 = None
    permute_253: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_174, [0, 3, 1, 2]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_271: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    amax_19: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_271, [-1], True)
    sub_58: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_271, amax_19);  clone_271 = amax_19 = None
    exp_19: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_20: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_254: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_19, [0, 2, 3, 1]);  div_19 = None
    permute_255: "f32[16, 16]" = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
    clone_272: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_389: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_272, [2654208, 16]);  clone_272 = None
    mm_39: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_389, permute_255);  view_389 = permute_255 = None
    view_390: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_39, [8, 576, 576, 16]);  mm_39 = None
    add_175: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_390, arg391_1);  view_390 = arg391_1 = None
    permute_256: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_175, [0, 3, 1, 2]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_273: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_256);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_78: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_273, [8, 16, 576, 576]);  clone_273 = None
    clone_274: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_391: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_274, [128, 576, 576]);  clone_274 = None
    expand_79: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_59, [8, 16, 576, 48]);  select_59 = None
    clone_275: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
    view_392: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_275, [128, 576, 48]);  clone_275 = None
    bmm_39: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_391, view_392);  view_391 = view_392 = None
    view_393: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_39, [8, 16, 576, 48]);  bmm_39 = None
    permute_257: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    clone_276: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    view_394: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_276, [8, 576, 768]);  clone_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_395: "f32[4608, 768]" = torch.ops.aten.view.default(view_394, [4608, 768]);  view_394 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
    addmm_77: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg393_1, view_395, permute_258);  arg393_1 = view_395 = permute_258 = None
    view_396: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_77, [8, 576, 768]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_277: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_396);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_193: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg39_1, clone_277);  arg39_1 = clone_277 = None
    add_176: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_171, mul_193);  add_171 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_278: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_176, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_278, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 576, 1]" = var_mean_39[0]
    getitem_79: "f32[8, 576, 1]" = var_mean_39[1];  var_mean_39 = None
    add_177: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_39: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_59: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_278, getitem_79);  clone_278 = getitem_79 = None
    mul_194: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
    mul_195: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_194, arg394_1);  mul_194 = arg394_1 = None
    add_178: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_195, arg395_1);  mul_195 = arg395_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_397: "f32[4608, 768]" = torch.ops.aten.view.default(add_178, [4608, 768]);  add_178 = None
    permute_259: "f32[768, 3072]" = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
    addmm_78: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg397_1, view_397, permute_259);  arg397_1 = view_397 = permute_259 = None
    view_398: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_78, [8, 576, 3072]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_196: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_398, 0.5)
    mul_197: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_398, 0.7071067811865476);  view_398 = None
    erf_19: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
    add_179: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_198: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_196, add_179);  mul_196 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_279: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_198);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_399: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_279, [4608, 3072]);  clone_279 = None
    permute_260: "f32[3072, 768]" = torch.ops.aten.permute.default(arg398_1, [1, 0]);  arg398_1 = None
    addmm_79: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg399_1, view_399, permute_260);  arg399_1 = view_399 = permute_260 = None
    view_400: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_79, [8, 576, 768]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_280: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_400);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_199: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg40_1, clone_280);  arg40_1 = clone_280 = None
    add_180: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_176, mul_199);  add_176 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_281: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_180, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_281, [2], correction = 0, keepdim = True)
    getitem_80: "f32[8, 576, 1]" = var_mean_40[0]
    getitem_81: "f32[8, 576, 1]" = var_mean_40[1];  var_mean_40 = None
    add_181: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_40: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_60: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_281, getitem_81);  clone_281 = getitem_81 = None
    mul_200: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
    mul_201: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_200, arg400_1);  mul_200 = arg400_1 = None
    add_182: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_201, arg401_1);  mul_201 = arg401_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_401: "f32[4608, 768]" = torch.ops.aten.view.default(add_182, [4608, 768]);  add_182 = None
    permute_261: "f32[768, 2304]" = torch.ops.aten.permute.default(arg402_1, [1, 0]);  arg402_1 = None
    addmm_80: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg403_1, view_401, permute_261);  arg403_1 = view_401 = permute_261 = None
    view_402: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_80, [8, 576, 2304]);  addmm_80 = None
    view_403: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_402, [8, 576, 3, 16, 48]);  view_402 = None
    permute_262: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_403, [2, 0, 3, 1, 4]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_60: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_262, 0, 0)
    mul_202: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_60, 0.14433756729740643);  select_60 = None
    select_61: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_262, 0, 1)
    select_62: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_262, 0, 2);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_263: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_61, [0, 1, 3, 2]);  select_61 = None
    expand_80: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_202, [8, 16, 576, 48]);  mul_202 = None
    clone_282: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_404: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_282, [128, 576, 48]);  clone_282 = None
    expand_81: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_263, [8, 16, 48, 576]);  permute_263 = None
    clone_283: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_405: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_283, [128, 48, 576]);  clone_283 = None
    bmm_40: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_404, view_405);  view_404 = view_405 = None
    view_406: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_40, [8, 16, 576, 576]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_264: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_406, [0, 2, 3, 1]);  view_406 = None
    permute_265: "f32[16, 16]" = torch.ops.aten.permute.default(arg404_1, [1, 0]);  arg404_1 = None
    clone_284: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
    view_407: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_284, [2654208, 16]);  clone_284 = None
    mm_40: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_407, permute_265);  view_407 = permute_265 = None
    view_408: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_40, [8, 576, 576, 16]);  mm_40 = None
    add_183: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_408, arg405_1);  view_408 = arg405_1 = None
    permute_266: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_183, [0, 3, 1, 2]);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_285: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    amax_20: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_285, [-1], True)
    sub_61: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_285, amax_20);  clone_285 = amax_20 = None
    exp_20: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_21: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_267: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_20, [0, 2, 3, 1]);  div_20 = None
    permute_268: "f32[16, 16]" = torch.ops.aten.permute.default(arg406_1, [1, 0]);  arg406_1 = None
    clone_286: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_409: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_286, [2654208, 16]);  clone_286 = None
    mm_41: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_409, permute_268);  view_409 = permute_268 = None
    view_410: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_41, [8, 576, 576, 16]);  mm_41 = None
    add_184: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_410, arg407_1);  view_410 = arg407_1 = None
    permute_269: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_184, [0, 3, 1, 2]);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_287: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_269);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_82: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_287, [8, 16, 576, 576]);  clone_287 = None
    clone_288: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
    view_411: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_288, [128, 576, 576]);  clone_288 = None
    expand_83: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_62, [8, 16, 576, 48]);  select_62 = None
    clone_289: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_412: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_289, [128, 576, 48]);  clone_289 = None
    bmm_41: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_411, view_412);  view_411 = view_412 = None
    view_413: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_41, [8, 16, 576, 48]);  bmm_41 = None
    permute_270: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    clone_290: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
    view_414: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_290, [8, 576, 768]);  clone_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_415: "f32[4608, 768]" = torch.ops.aten.view.default(view_414, [4608, 768]);  view_414 = None
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(arg408_1, [1, 0]);  arg408_1 = None
    addmm_81: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg409_1, view_415, permute_271);  arg409_1 = view_415 = permute_271 = None
    view_416: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_81, [8, 576, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_291: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_416);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_203: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg41_1, clone_291);  arg41_1 = clone_291 = None
    add_185: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_180, mul_203);  add_180 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_292: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_185, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_292, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 576, 1]" = var_mean_41[0]
    getitem_83: "f32[8, 576, 1]" = var_mean_41[1];  var_mean_41 = None
    add_186: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
    rsqrt_41: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_62: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_292, getitem_83);  clone_292 = getitem_83 = None
    mul_204: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
    mul_205: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_204, arg410_1);  mul_204 = arg410_1 = None
    add_187: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_205, arg411_1);  mul_205 = arg411_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_417: "f32[4608, 768]" = torch.ops.aten.view.default(add_187, [4608, 768]);  add_187 = None
    permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(arg412_1, [1, 0]);  arg412_1 = None
    addmm_82: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg413_1, view_417, permute_272);  arg413_1 = view_417 = permute_272 = None
    view_418: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_82, [8, 576, 3072]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_206: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_418, 0.5)
    mul_207: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_418, 0.7071067811865476);  view_418 = None
    erf_20: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_188: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_208: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_206, add_188);  mul_206 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_293: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_208);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_419: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_293, [4608, 3072]);  clone_293 = None
    permute_273: "f32[3072, 768]" = torch.ops.aten.permute.default(arg414_1, [1, 0]);  arg414_1 = None
    addmm_83: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg415_1, view_419, permute_273);  arg415_1 = view_419 = permute_273 = None
    view_420: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_83, [8, 576, 768]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_294: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_420);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_209: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg42_1, clone_294);  arg42_1 = clone_294 = None
    add_189: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_185, mul_209);  add_185 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_295: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_189, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_295, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 576, 1]" = var_mean_42[0]
    getitem_85: "f32[8, 576, 1]" = var_mean_42[1];  var_mean_42 = None
    add_190: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_42: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    sub_63: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_295, getitem_85);  clone_295 = getitem_85 = None
    mul_210: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
    mul_211: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_210, arg416_1);  mul_210 = arg416_1 = None
    add_191: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_211, arg417_1);  mul_211 = arg417_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_421: "f32[4608, 768]" = torch.ops.aten.view.default(add_191, [4608, 768]);  add_191 = None
    permute_274: "f32[768, 2304]" = torch.ops.aten.permute.default(arg418_1, [1, 0]);  arg418_1 = None
    addmm_84: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg419_1, view_421, permute_274);  arg419_1 = view_421 = permute_274 = None
    view_422: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_84, [8, 576, 2304]);  addmm_84 = None
    view_423: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_422, [8, 576, 3, 16, 48]);  view_422 = None
    permute_275: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_423, [2, 0, 3, 1, 4]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_63: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_275, 0, 0)
    mul_212: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_63, 0.14433756729740643);  select_63 = None
    select_64: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_275, 0, 1)
    select_65: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_275, 0, 2);  permute_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_276: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_64, [0, 1, 3, 2]);  select_64 = None
    expand_84: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_212, [8, 16, 576, 48]);  mul_212 = None
    clone_296: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_424: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_296, [128, 576, 48]);  clone_296 = None
    expand_85: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_276, [8, 16, 48, 576]);  permute_276 = None
    clone_297: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_425: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_297, [128, 48, 576]);  clone_297 = None
    bmm_42: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_424, view_425);  view_424 = view_425 = None
    view_426: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_42, [8, 16, 576, 576]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_277: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_426, [0, 2, 3, 1]);  view_426 = None
    permute_278: "f32[16, 16]" = torch.ops.aten.permute.default(arg420_1, [1, 0]);  arg420_1 = None
    clone_298: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    view_427: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_298, [2654208, 16]);  clone_298 = None
    mm_42: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_427, permute_278);  view_427 = permute_278 = None
    view_428: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_42, [8, 576, 576, 16]);  mm_42 = None
    add_192: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_428, arg421_1);  view_428 = arg421_1 = None
    permute_279: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_192, [0, 3, 1, 2]);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_299: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    amax_21: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_299, [-1], True)
    sub_64: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_299, amax_21);  clone_299 = amax_21 = None
    exp_21: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_22: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_280: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_21, [0, 2, 3, 1]);  div_21 = None
    permute_281: "f32[16, 16]" = torch.ops.aten.permute.default(arg422_1, [1, 0]);  arg422_1 = None
    clone_300: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_280, memory_format = torch.contiguous_format);  permute_280 = None
    view_429: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_300, [2654208, 16]);  clone_300 = None
    mm_43: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_429, permute_281);  view_429 = permute_281 = None
    view_430: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_43, [8, 576, 576, 16]);  mm_43 = None
    add_193: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_430, arg423_1);  view_430 = arg423_1 = None
    permute_282: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_193, [0, 3, 1, 2]);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_301: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_282);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_86: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_301, [8, 16, 576, 576]);  clone_301 = None
    clone_302: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    view_431: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_302, [128, 576, 576]);  clone_302 = None
    expand_87: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_65, [8, 16, 576, 48]);  select_65 = None
    clone_303: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    view_432: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_303, [128, 576, 48]);  clone_303 = None
    bmm_43: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_431, view_432);  view_431 = view_432 = None
    view_433: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_43, [8, 16, 576, 48]);  bmm_43 = None
    permute_283: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None
    clone_304: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    view_434: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_304, [8, 576, 768]);  clone_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_435: "f32[4608, 768]" = torch.ops.aten.view.default(view_434, [4608, 768]);  view_434 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(arg424_1, [1, 0]);  arg424_1 = None
    addmm_85: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg425_1, view_435, permute_284);  arg425_1 = view_435 = permute_284 = None
    view_436: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_85, [8, 576, 768]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_305: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_436);  view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_213: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg43_1, clone_305);  arg43_1 = clone_305 = None
    add_194: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_189, mul_213);  add_189 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_306: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_194, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_306, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 576, 1]" = var_mean_43[0]
    getitem_87: "f32[8, 576, 1]" = var_mean_43[1];  var_mean_43 = None
    add_195: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_43: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    sub_65: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_306, getitem_87);  clone_306 = getitem_87 = None
    mul_214: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
    mul_215: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_214, arg426_1);  mul_214 = arg426_1 = None
    add_196: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_215, arg427_1);  mul_215 = arg427_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_437: "f32[4608, 768]" = torch.ops.aten.view.default(add_196, [4608, 768]);  add_196 = None
    permute_285: "f32[768, 3072]" = torch.ops.aten.permute.default(arg428_1, [1, 0]);  arg428_1 = None
    addmm_86: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg429_1, view_437, permute_285);  arg429_1 = view_437 = permute_285 = None
    view_438: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_86, [8, 576, 3072]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_216: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_438, 0.5)
    mul_217: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_438, 0.7071067811865476);  view_438 = None
    erf_21: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_217);  mul_217 = None
    add_197: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_218: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_216, add_197);  mul_216 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_307: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_218);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_439: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_307, [4608, 3072]);  clone_307 = None
    permute_286: "f32[3072, 768]" = torch.ops.aten.permute.default(arg430_1, [1, 0]);  arg430_1 = None
    addmm_87: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg431_1, view_439, permute_286);  arg431_1 = view_439 = permute_286 = None
    view_440: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_87, [8, 576, 768]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_308: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_440);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_219: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg44_1, clone_308);  arg44_1 = clone_308 = None
    add_198: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_194, mul_219);  add_194 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_309: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_198, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_309, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 576, 1]" = var_mean_44[0]
    getitem_89: "f32[8, 576, 1]" = var_mean_44[1];  var_mean_44 = None
    add_199: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_44: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_66: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_309, getitem_89);  clone_309 = getitem_89 = None
    mul_220: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
    mul_221: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_220, arg432_1);  mul_220 = arg432_1 = None
    add_200: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_221, arg433_1);  mul_221 = arg433_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_441: "f32[4608, 768]" = torch.ops.aten.view.default(add_200, [4608, 768]);  add_200 = None
    permute_287: "f32[768, 2304]" = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
    addmm_88: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg435_1, view_441, permute_287);  arg435_1 = view_441 = permute_287 = None
    view_442: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_88, [8, 576, 2304]);  addmm_88 = None
    view_443: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_442, [8, 576, 3, 16, 48]);  view_442 = None
    permute_288: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_443, [2, 0, 3, 1, 4]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_66: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_288, 0, 0)
    mul_222: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_66, 0.14433756729740643);  select_66 = None
    select_67: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_288, 0, 1)
    select_68: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_288, 0, 2);  permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_289: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_67, [0, 1, 3, 2]);  select_67 = None
    expand_88: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_222, [8, 16, 576, 48]);  mul_222 = None
    clone_310: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_444: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_310, [128, 576, 48]);  clone_310 = None
    expand_89: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_289, [8, 16, 48, 576]);  permute_289 = None
    clone_311: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_445: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_311, [128, 48, 576]);  clone_311 = None
    bmm_44: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_444, view_445);  view_444 = view_445 = None
    view_446: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_44, [8, 16, 576, 576]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_290: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_446, [0, 2, 3, 1]);  view_446 = None
    permute_291: "f32[16, 16]" = torch.ops.aten.permute.default(arg436_1, [1, 0]);  arg436_1 = None
    clone_312: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
    view_447: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_312, [2654208, 16]);  clone_312 = None
    mm_44: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_447, permute_291);  view_447 = permute_291 = None
    view_448: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_44, [8, 576, 576, 16]);  mm_44 = None
    add_201: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_448, arg437_1);  view_448 = arg437_1 = None
    permute_292: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_201, [0, 3, 1, 2]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_313: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    amax_22: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_313, [-1], True)
    sub_67: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_313, amax_22);  clone_313 = amax_22 = None
    exp_22: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_23: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_293: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_22, [0, 2, 3, 1]);  div_22 = None
    permute_294: "f32[16, 16]" = torch.ops.aten.permute.default(arg438_1, [1, 0]);  arg438_1 = None
    clone_314: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_449: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_314, [2654208, 16]);  clone_314 = None
    mm_45: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_449, permute_294);  view_449 = permute_294 = None
    view_450: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_45, [8, 576, 576, 16]);  mm_45 = None
    add_202: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_450, arg439_1);  view_450 = arg439_1 = None
    permute_295: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_202, [0, 3, 1, 2]);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_315: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_295);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_90: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_315, [8, 16, 576, 576]);  clone_315 = None
    clone_316: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
    view_451: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_316, [128, 576, 576]);  clone_316 = None
    expand_91: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_68, [8, 16, 576, 48]);  select_68 = None
    clone_317: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
    view_452: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_317, [128, 576, 48]);  clone_317 = None
    bmm_45: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_451, view_452);  view_451 = view_452 = None
    view_453: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_45, [8, 16, 576, 48]);  bmm_45 = None
    permute_296: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    clone_318: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    view_454: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_318, [8, 576, 768]);  clone_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_455: "f32[4608, 768]" = torch.ops.aten.view.default(view_454, [4608, 768]);  view_454 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(arg440_1, [1, 0]);  arg440_1 = None
    addmm_89: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg441_1, view_455, permute_297);  arg441_1 = view_455 = permute_297 = None
    view_456: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_89, [8, 576, 768]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_319: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_456);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_223: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg45_1, clone_319);  arg45_1 = clone_319 = None
    add_203: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_198, mul_223);  add_198 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_320: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_203, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_320, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 576, 1]" = var_mean_45[0]
    getitem_91: "f32[8, 576, 1]" = var_mean_45[1];  var_mean_45 = None
    add_204: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_45: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_68: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_320, getitem_91);  clone_320 = getitem_91 = None
    mul_224: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
    mul_225: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_224, arg442_1);  mul_224 = arg442_1 = None
    add_205: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_225, arg443_1);  mul_225 = arg443_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_457: "f32[4608, 768]" = torch.ops.aten.view.default(add_205, [4608, 768]);  add_205 = None
    permute_298: "f32[768, 3072]" = torch.ops.aten.permute.default(arg444_1, [1, 0]);  arg444_1 = None
    addmm_90: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg445_1, view_457, permute_298);  arg445_1 = view_457 = permute_298 = None
    view_458: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_90, [8, 576, 3072]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_226: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_458, 0.5)
    mul_227: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_458, 0.7071067811865476);  view_458 = None
    erf_22: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_206: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_228: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_226, add_206);  mul_226 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_321: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_228);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_459: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_321, [4608, 3072]);  clone_321 = None
    permute_299: "f32[3072, 768]" = torch.ops.aten.permute.default(arg446_1, [1, 0]);  arg446_1 = None
    addmm_91: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg447_1, view_459, permute_299);  arg447_1 = view_459 = permute_299 = None
    view_460: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_91, [8, 576, 768]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_322: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_460);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_229: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg46_1, clone_322);  arg46_1 = clone_322 = None
    add_207: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_203, mul_229);  add_203 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_323: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_207, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_323, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 576, 1]" = var_mean_46[0]
    getitem_93: "f32[8, 576, 1]" = var_mean_46[1];  var_mean_46 = None
    add_208: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_46: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_69: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_323, getitem_93);  clone_323 = getitem_93 = None
    mul_230: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
    mul_231: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_230, arg448_1);  mul_230 = arg448_1 = None
    add_209: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_231, arg449_1);  mul_231 = arg449_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_461: "f32[4608, 768]" = torch.ops.aten.view.default(add_209, [4608, 768]);  add_209 = None
    permute_300: "f32[768, 2304]" = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
    addmm_92: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg451_1, view_461, permute_300);  arg451_1 = view_461 = permute_300 = None
    view_462: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_92, [8, 576, 2304]);  addmm_92 = None
    view_463: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_462, [8, 576, 3, 16, 48]);  view_462 = None
    permute_301: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_463, [2, 0, 3, 1, 4]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_69: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_301, 0, 0)
    mul_232: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_69, 0.14433756729740643);  select_69 = None
    select_70: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_301, 0, 1)
    select_71: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_301, 0, 2);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_302: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_70, [0, 1, 3, 2]);  select_70 = None
    expand_92: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_232, [8, 16, 576, 48]);  mul_232 = None
    clone_324: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_464: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_324, [128, 576, 48]);  clone_324 = None
    expand_93: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_302, [8, 16, 48, 576]);  permute_302 = None
    clone_325: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_465: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_325, [128, 48, 576]);  clone_325 = None
    bmm_46: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_464, view_465);  view_464 = view_465 = None
    view_466: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_46, [8, 16, 576, 576]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_303: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_466, [0, 2, 3, 1]);  view_466 = None
    permute_304: "f32[16, 16]" = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
    clone_326: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    view_467: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_326, [2654208, 16]);  clone_326 = None
    mm_46: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_467, permute_304);  view_467 = permute_304 = None
    view_468: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_46, [8, 576, 576, 16]);  mm_46 = None
    add_210: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_468, arg453_1);  view_468 = arg453_1 = None
    permute_305: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_210, [0, 3, 1, 2]);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_327: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    amax_23: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_327, [-1], True)
    sub_70: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_327, amax_23);  clone_327 = amax_23 = None
    exp_23: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_24: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_306: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_23, [0, 2, 3, 1]);  div_23 = None
    permute_307: "f32[16, 16]" = torch.ops.aten.permute.default(arg454_1, [1, 0]);  arg454_1 = None
    clone_328: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    view_469: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_328, [2654208, 16]);  clone_328 = None
    mm_47: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_469, permute_307);  view_469 = permute_307 = None
    view_470: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_47, [8, 576, 576, 16]);  mm_47 = None
    add_211: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_470, arg455_1);  view_470 = arg455_1 = None
    permute_308: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_211, [0, 3, 1, 2]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_329: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_308);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_94: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_329, [8, 16, 576, 576]);  clone_329 = None
    clone_330: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
    view_471: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_330, [128, 576, 576]);  clone_330 = None
    expand_95: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_71, [8, 16, 576, 48]);  select_71 = None
    clone_331: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    view_472: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_331, [128, 576, 48]);  clone_331 = None
    bmm_47: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_471, view_472);  view_471 = view_472 = None
    view_473: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_47, [8, 16, 576, 48]);  bmm_47 = None
    permute_309: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None
    clone_332: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    view_474: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_332, [8, 576, 768]);  clone_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_475: "f32[4608, 768]" = torch.ops.aten.view.default(view_474, [4608, 768]);  view_474 = None
    permute_310: "f32[768, 768]" = torch.ops.aten.permute.default(arg456_1, [1, 0]);  arg456_1 = None
    addmm_93: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg457_1, view_475, permute_310);  arg457_1 = view_475 = permute_310 = None
    view_476: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_93, [8, 576, 768]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_333: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_476);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_233: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg47_1, clone_333);  arg47_1 = clone_333 = None
    add_212: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_207, mul_233);  add_207 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_334: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_212, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_334, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 576, 1]" = var_mean_47[0]
    getitem_95: "f32[8, 576, 1]" = var_mean_47[1];  var_mean_47 = None
    add_213: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
    rsqrt_47: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_71: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_334, getitem_95);  clone_334 = getitem_95 = None
    mul_234: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
    mul_235: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_234, arg458_1);  mul_234 = arg458_1 = None
    add_214: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_235, arg459_1);  mul_235 = arg459_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_477: "f32[4608, 768]" = torch.ops.aten.view.default(add_214, [4608, 768]);  add_214 = None
    permute_311: "f32[768, 3072]" = torch.ops.aten.permute.default(arg460_1, [1, 0]);  arg460_1 = None
    addmm_94: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg461_1, view_477, permute_311);  arg461_1 = view_477 = permute_311 = None
    view_478: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_94, [8, 576, 3072]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_236: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_478, 0.5)
    mul_237: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_478, 0.7071067811865476);  view_478 = None
    erf_23: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_237);  mul_237 = None
    add_215: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_238: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_236, add_215);  mul_236 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_335: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_238);  mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_479: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_335, [4608, 3072]);  clone_335 = None
    permute_312: "f32[3072, 768]" = torch.ops.aten.permute.default(arg462_1, [1, 0]);  arg462_1 = None
    addmm_95: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg463_1, view_479, permute_312);  arg463_1 = view_479 = permute_312 = None
    view_480: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_95, [8, 576, 768]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_336: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_480);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_239: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg48_1, clone_336);  arg48_1 = clone_336 = None
    add_216: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_212, mul_239);  add_212 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_337: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_216, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_337, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 576, 1]" = var_mean_48[0]
    getitem_97: "f32[8, 576, 1]" = var_mean_48[1];  var_mean_48 = None
    add_217: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_48: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    sub_72: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_337, getitem_97);  clone_337 = getitem_97 = None
    mul_240: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
    mul_241: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_240, arg464_1);  mul_240 = arg464_1 = None
    add_218: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_241, arg465_1);  mul_241 = arg465_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_481: "f32[4608, 768]" = torch.ops.aten.view.default(add_218, [4608, 768]);  add_218 = None
    permute_313: "f32[768, 2304]" = torch.ops.aten.permute.default(arg466_1, [1, 0]);  arg466_1 = None
    addmm_96: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg467_1, view_481, permute_313);  arg467_1 = view_481 = permute_313 = None
    view_482: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_96, [8, 576, 2304]);  addmm_96 = None
    view_483: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_482, [8, 576, 3, 16, 48]);  view_482 = None
    permute_314: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_483, [2, 0, 3, 1, 4]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_72: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_314, 0, 0)
    mul_242: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_72, 0.14433756729740643);  select_72 = None
    select_73: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_314, 0, 1)
    select_74: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_314, 0, 2);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_315: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_73, [0, 1, 3, 2]);  select_73 = None
    expand_96: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_242, [8, 16, 576, 48]);  mul_242 = None
    clone_338: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
    view_484: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_338, [128, 576, 48]);  clone_338 = None
    expand_97: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_315, [8, 16, 48, 576]);  permute_315 = None
    clone_339: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
    view_485: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_339, [128, 48, 576]);  clone_339 = None
    bmm_48: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_484, view_485);  view_484 = view_485 = None
    view_486: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_48, [8, 16, 576, 576]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_316: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_486, [0, 2, 3, 1]);  view_486 = None
    permute_317: "f32[16, 16]" = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
    clone_340: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    view_487: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_340, [2654208, 16]);  clone_340 = None
    mm_48: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_487, permute_317);  view_487 = permute_317 = None
    view_488: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_48, [8, 576, 576, 16]);  mm_48 = None
    add_219: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_488, arg469_1);  view_488 = arg469_1 = None
    permute_318: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_219, [0, 3, 1, 2]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_341: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
    amax_24: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_341, [-1], True)
    sub_73: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_341, amax_24);  clone_341 = amax_24 = None
    exp_24: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_25: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_24: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_319: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_24, [0, 2, 3, 1]);  div_24 = None
    permute_320: "f32[16, 16]" = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
    clone_342: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_489: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_342, [2654208, 16]);  clone_342 = None
    mm_49: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_489, permute_320);  view_489 = permute_320 = None
    view_490: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_49, [8, 576, 576, 16]);  mm_49 = None
    add_220: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_490, arg471_1);  view_490 = arg471_1 = None
    permute_321: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_220, [0, 3, 1, 2]);  add_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_343: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_321);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_98: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_343, [8, 16, 576, 576]);  clone_343 = None
    clone_344: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
    view_491: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_344, [128, 576, 576]);  clone_344 = None
    expand_99: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_74, [8, 16, 576, 48]);  select_74 = None
    clone_345: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
    view_492: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_345, [128, 576, 48]);  clone_345 = None
    bmm_49: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_491, view_492);  view_491 = view_492 = None
    view_493: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_49, [8, 16, 576, 48]);  bmm_49 = None
    permute_322: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    clone_346: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_494: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_346, [8, 576, 768]);  clone_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_495: "f32[4608, 768]" = torch.ops.aten.view.default(view_494, [4608, 768]);  view_494 = None
    permute_323: "f32[768, 768]" = torch.ops.aten.permute.default(arg472_1, [1, 0]);  arg472_1 = None
    addmm_97: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg473_1, view_495, permute_323);  arg473_1 = view_495 = permute_323 = None
    view_496: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_97, [8, 576, 768]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_347: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_496);  view_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_243: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg49_1, clone_347);  arg49_1 = clone_347 = None
    add_221: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_216, mul_243);  add_216 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_348: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_221, memory_format = torch.contiguous_format)
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_348, [2], correction = 0, keepdim = True)
    getitem_98: "f32[8, 576, 1]" = var_mean_49[0]
    getitem_99: "f32[8, 576, 1]" = var_mean_49[1];  var_mean_49 = None
    add_222: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
    rsqrt_49: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    sub_74: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_348, getitem_99);  clone_348 = getitem_99 = None
    mul_244: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = rsqrt_49 = None
    mul_245: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_244, arg474_1);  mul_244 = arg474_1 = None
    add_223: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_245, arg475_1);  mul_245 = arg475_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_497: "f32[4608, 768]" = torch.ops.aten.view.default(add_223, [4608, 768]);  add_223 = None
    permute_324: "f32[768, 3072]" = torch.ops.aten.permute.default(arg476_1, [1, 0]);  arg476_1 = None
    addmm_98: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg477_1, view_497, permute_324);  arg477_1 = view_497 = permute_324 = None
    view_498: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_98, [8, 576, 3072]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_246: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_498, 0.5)
    mul_247: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_498, 0.7071067811865476);  view_498 = None
    erf_24: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
    add_224: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_248: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_246, add_224);  mul_246 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_349: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_248);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_499: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_349, [4608, 3072]);  clone_349 = None
    permute_325: "f32[3072, 768]" = torch.ops.aten.permute.default(arg478_1, [1, 0]);  arg478_1 = None
    addmm_99: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg479_1, view_499, permute_325);  arg479_1 = view_499 = permute_325 = None
    view_500: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_99, [8, 576, 768]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_350: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_500);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_249: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg50_1, clone_350);  arg50_1 = clone_350 = None
    add_225: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_221, mul_249);  add_221 = mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_351: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_351, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 576, 1]" = var_mean_50[0]
    getitem_101: "f32[8, 576, 1]" = var_mean_50[1];  var_mean_50 = None
    add_226: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_50: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_75: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_351, getitem_101);  clone_351 = getitem_101 = None
    mul_250: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_50);  sub_75 = rsqrt_50 = None
    mul_251: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_250, arg480_1);  mul_250 = arg480_1 = None
    add_227: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_251, arg481_1);  mul_251 = arg481_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_501: "f32[4608, 768]" = torch.ops.aten.view.default(add_227, [4608, 768]);  add_227 = None
    permute_326: "f32[768, 2304]" = torch.ops.aten.permute.default(arg482_1, [1, 0]);  arg482_1 = None
    addmm_100: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg483_1, view_501, permute_326);  arg483_1 = view_501 = permute_326 = None
    view_502: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_100, [8, 576, 2304]);  addmm_100 = None
    view_503: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_502, [8, 576, 3, 16, 48]);  view_502 = None
    permute_327: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_503, [2, 0, 3, 1, 4]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_75: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_327, 0, 0)
    mul_252: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_75, 0.14433756729740643);  select_75 = None
    select_76: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_327, 0, 1)
    select_77: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_327, 0, 2);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_328: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_76, [0, 1, 3, 2]);  select_76 = None
    expand_100: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_252, [8, 16, 576, 48]);  mul_252 = None
    clone_352: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
    view_504: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_352, [128, 576, 48]);  clone_352 = None
    expand_101: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_328, [8, 16, 48, 576]);  permute_328 = None
    clone_353: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
    view_505: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_353, [128, 48, 576]);  clone_353 = None
    bmm_50: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_504, view_505);  view_504 = view_505 = None
    view_506: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_50, [8, 16, 576, 576]);  bmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_329: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_506, [0, 2, 3, 1]);  view_506 = None
    permute_330: "f32[16, 16]" = torch.ops.aten.permute.default(arg484_1, [1, 0]);  arg484_1 = None
    clone_354: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
    view_507: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_354, [2654208, 16]);  clone_354 = None
    mm_50: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_507, permute_330);  view_507 = permute_330 = None
    view_508: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_50, [8, 576, 576, 16]);  mm_50 = None
    add_228: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_508, arg485_1);  view_508 = arg485_1 = None
    permute_331: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_228, [0, 3, 1, 2]);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_355: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    amax_25: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_355, [-1], True)
    sub_76: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_355, amax_25);  clone_355 = amax_25 = None
    exp_25: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_26: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
    div_25: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_332: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_25, [0, 2, 3, 1]);  div_25 = None
    permute_333: "f32[16, 16]" = torch.ops.aten.permute.default(arg486_1, [1, 0]);  arg486_1 = None
    clone_356: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_332, memory_format = torch.contiguous_format);  permute_332 = None
    view_509: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_356, [2654208, 16]);  clone_356 = None
    mm_51: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_509, permute_333);  view_509 = permute_333 = None
    view_510: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_51, [8, 576, 576, 16]);  mm_51 = None
    add_229: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_510, arg487_1);  view_510 = arg487_1 = None
    permute_334: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_229, [0, 3, 1, 2]);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_357: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_334);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_102: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_357, [8, 16, 576, 576]);  clone_357 = None
    clone_358: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
    view_511: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_358, [128, 576, 576]);  clone_358 = None
    expand_103: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_77, [8, 16, 576, 48]);  select_77 = None
    clone_359: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
    view_512: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_359, [128, 576, 48]);  clone_359 = None
    bmm_51: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_511, view_512);  view_511 = view_512 = None
    view_513: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_51, [8, 16, 576, 48]);  bmm_51 = None
    permute_335: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
    clone_360: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
    view_514: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_360, [8, 576, 768]);  clone_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_515: "f32[4608, 768]" = torch.ops.aten.view.default(view_514, [4608, 768]);  view_514 = None
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(arg488_1, [1, 0]);  arg488_1 = None
    addmm_101: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg489_1, view_515, permute_336);  arg489_1 = view_515 = permute_336 = None
    view_516: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_101, [8, 576, 768]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_361: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_516);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_253: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg51_1, clone_361);  arg51_1 = clone_361 = None
    add_230: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_225, mul_253);  add_225 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_362: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_230, memory_format = torch.contiguous_format)
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_362, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 576, 1]" = var_mean_51[0]
    getitem_103: "f32[8, 576, 1]" = var_mean_51[1];  var_mean_51 = None
    add_231: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_51: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_77: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_362, getitem_103);  clone_362 = getitem_103 = None
    mul_254: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_51);  sub_77 = rsqrt_51 = None
    mul_255: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_254, arg490_1);  mul_254 = arg490_1 = None
    add_232: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_255, arg491_1);  mul_255 = arg491_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_517: "f32[4608, 768]" = torch.ops.aten.view.default(add_232, [4608, 768]);  add_232 = None
    permute_337: "f32[768, 3072]" = torch.ops.aten.permute.default(arg492_1, [1, 0]);  arg492_1 = None
    addmm_102: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg493_1, view_517, permute_337);  arg493_1 = view_517 = permute_337 = None
    view_518: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_102, [8, 576, 3072]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_256: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_518, 0.5)
    mul_257: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_518, 0.7071067811865476);  view_518 = None
    erf_25: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_257);  mul_257 = None
    add_233: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_258: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_256, add_233);  mul_256 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_363: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_258);  mul_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_519: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_363, [4608, 3072]);  clone_363 = None
    permute_338: "f32[3072, 768]" = torch.ops.aten.permute.default(arg494_1, [1, 0]);  arg494_1 = None
    addmm_103: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg495_1, view_519, permute_338);  arg495_1 = view_519 = permute_338 = None
    view_520: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_103, [8, 576, 768]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_364: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_520);  view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_259: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg52_1, clone_364);  arg52_1 = clone_364 = None
    add_234: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_230, mul_259);  add_230 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_365: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_234, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_365, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 576, 1]" = var_mean_52[0]
    getitem_105: "f32[8, 576, 1]" = var_mean_52[1];  var_mean_52 = None
    add_235: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_52: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_78: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_365, getitem_105);  clone_365 = getitem_105 = None
    mul_260: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_52);  sub_78 = rsqrt_52 = None
    mul_261: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_260, arg496_1);  mul_260 = arg496_1 = None
    add_236: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_261, arg497_1);  mul_261 = arg497_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_521: "f32[4608, 768]" = torch.ops.aten.view.default(add_236, [4608, 768]);  add_236 = None
    permute_339: "f32[768, 2304]" = torch.ops.aten.permute.default(arg498_1, [1, 0]);  arg498_1 = None
    addmm_104: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg499_1, view_521, permute_339);  arg499_1 = view_521 = permute_339 = None
    view_522: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_104, [8, 576, 2304]);  addmm_104 = None
    view_523: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_522, [8, 576, 3, 16, 48]);  view_522 = None
    permute_340: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_523, [2, 0, 3, 1, 4]);  view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_78: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_340, 0, 0)
    mul_262: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_78, 0.14433756729740643);  select_78 = None
    select_79: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_340, 0, 1)
    select_80: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_340, 0, 2);  permute_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_341: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_79, [0, 1, 3, 2]);  select_79 = None
    expand_104: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_262, [8, 16, 576, 48]);  mul_262 = None
    clone_366: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    view_524: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_366, [128, 576, 48]);  clone_366 = None
    expand_105: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_341, [8, 16, 48, 576]);  permute_341 = None
    clone_367: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    view_525: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_367, [128, 48, 576]);  clone_367 = None
    bmm_52: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_524, view_525);  view_524 = view_525 = None
    view_526: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_52, [8, 16, 576, 576]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_342: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_526, [0, 2, 3, 1]);  view_526 = None
    permute_343: "f32[16, 16]" = torch.ops.aten.permute.default(arg500_1, [1, 0]);  arg500_1 = None
    clone_368: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
    view_527: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_368, [2654208, 16]);  clone_368 = None
    mm_52: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_527, permute_343);  view_527 = permute_343 = None
    view_528: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_52, [8, 576, 576, 16]);  mm_52 = None
    add_237: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_528, arg501_1);  view_528 = arg501_1 = None
    permute_344: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_237, [0, 3, 1, 2]);  add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_369: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
    amax_26: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_369, [-1], True)
    sub_79: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_369, amax_26);  clone_369 = amax_26 = None
    exp_26: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
    sum_27: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_26: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_345: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_26, [0, 2, 3, 1]);  div_26 = None
    permute_346: "f32[16, 16]" = torch.ops.aten.permute.default(arg502_1, [1, 0]);  arg502_1 = None
    clone_370: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    view_529: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_370, [2654208, 16]);  clone_370 = None
    mm_53: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_529, permute_346);  view_529 = permute_346 = None
    view_530: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_53, [8, 576, 576, 16]);  mm_53 = None
    add_238: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_530, arg503_1);  view_530 = arg503_1 = None
    permute_347: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_238, [0, 3, 1, 2]);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_371: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_347);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_106: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_371, [8, 16, 576, 576]);  clone_371 = None
    clone_372: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
    view_531: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_372, [128, 576, 576]);  clone_372 = None
    expand_107: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_80, [8, 16, 576, 48]);  select_80 = None
    clone_373: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
    view_532: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_373, [128, 576, 48]);  clone_373 = None
    bmm_53: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_531, view_532);  view_531 = view_532 = None
    view_533: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_53, [8, 16, 576, 48]);  bmm_53 = None
    permute_348: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
    clone_374: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_534: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_374, [8, 576, 768]);  clone_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_535: "f32[4608, 768]" = torch.ops.aten.view.default(view_534, [4608, 768]);  view_534 = None
    permute_349: "f32[768, 768]" = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
    addmm_105: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg505_1, view_535, permute_349);  arg505_1 = view_535 = permute_349 = None
    view_536: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_105, [8, 576, 768]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_375: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_536);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_263: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg53_1, clone_375);  arg53_1 = clone_375 = None
    add_239: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_234, mul_263);  add_234 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_376: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_376, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 576, 1]" = var_mean_53[0]
    getitem_107: "f32[8, 576, 1]" = var_mean_53[1];  var_mean_53 = None
    add_240: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_53: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    sub_80: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_376, getitem_107);  clone_376 = getitem_107 = None
    mul_264: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_53);  sub_80 = rsqrt_53 = None
    mul_265: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_264, arg506_1);  mul_264 = arg506_1 = None
    add_241: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_265, arg507_1);  mul_265 = arg507_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_537: "f32[4608, 768]" = torch.ops.aten.view.default(add_241, [4608, 768]);  add_241 = None
    permute_350: "f32[768, 3072]" = torch.ops.aten.permute.default(arg508_1, [1, 0]);  arg508_1 = None
    addmm_106: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg509_1, view_537, permute_350);  arg509_1 = view_537 = permute_350 = None
    view_538: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_106, [8, 576, 3072]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_266: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_538, 0.5)
    mul_267: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_538, 0.7071067811865476);  view_538 = None
    erf_26: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
    add_242: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_268: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_266, add_242);  mul_266 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_377: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_268);  mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_539: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_377, [4608, 3072]);  clone_377 = None
    permute_351: "f32[3072, 768]" = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
    addmm_107: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg511_1, view_539, permute_351);  arg511_1 = view_539 = permute_351 = None
    view_540: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_107, [8, 576, 768]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_378: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_540);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_269: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg54_1, clone_378);  arg54_1 = clone_378 = None
    add_243: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_239, mul_269);  add_239 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_379: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_243, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_379, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 576, 1]" = var_mean_54[0]
    getitem_109: "f32[8, 576, 1]" = var_mean_54[1];  var_mean_54 = None
    add_244: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_54: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_81: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_379, getitem_109);  clone_379 = getitem_109 = None
    mul_270: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_54);  sub_81 = rsqrt_54 = None
    mul_271: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_270, arg512_1);  mul_270 = arg512_1 = None
    add_245: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_271, arg513_1);  mul_271 = arg513_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_541: "f32[4608, 768]" = torch.ops.aten.view.default(add_245, [4608, 768]);  add_245 = None
    permute_352: "f32[768, 2304]" = torch.ops.aten.permute.default(arg514_1, [1, 0]);  arg514_1 = None
    addmm_108: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg515_1, view_541, permute_352);  arg515_1 = view_541 = permute_352 = None
    view_542: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_108, [8, 576, 2304]);  addmm_108 = None
    view_543: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_542, [8, 576, 3, 16, 48]);  view_542 = None
    permute_353: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_543, [2, 0, 3, 1, 4]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_81: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_353, 0, 0)
    mul_272: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_81, 0.14433756729740643);  select_81 = None
    select_82: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_353, 0, 1)
    select_83: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_353, 0, 2);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_354: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_82, [0, 1, 3, 2]);  select_82 = None
    expand_108: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_272, [8, 16, 576, 48]);  mul_272 = None
    clone_380: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
    view_544: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_380, [128, 576, 48]);  clone_380 = None
    expand_109: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_354, [8, 16, 48, 576]);  permute_354 = None
    clone_381: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
    view_545: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_381, [128, 48, 576]);  clone_381 = None
    bmm_54: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_544, view_545);  view_544 = view_545 = None
    view_546: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_54, [8, 16, 576, 576]);  bmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_355: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_546, [0, 2, 3, 1]);  view_546 = None
    permute_356: "f32[16, 16]" = torch.ops.aten.permute.default(arg516_1, [1, 0]);  arg516_1 = None
    clone_382: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_547: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_382, [2654208, 16]);  clone_382 = None
    mm_54: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_547, permute_356);  view_547 = permute_356 = None
    view_548: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_54, [8, 576, 576, 16]);  mm_54 = None
    add_246: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_548, arg517_1);  view_548 = arg517_1 = None
    permute_357: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_246, [0, 3, 1, 2]);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_383: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    amax_27: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_383, [-1], True)
    sub_82: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_383, amax_27);  clone_383 = amax_27 = None
    exp_27: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_82);  sub_82 = None
    sum_28: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
    div_27: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_358: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_27, [0, 2, 3, 1]);  div_27 = None
    permute_359: "f32[16, 16]" = torch.ops.aten.permute.default(arg518_1, [1, 0]);  arg518_1 = None
    clone_384: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_549: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_384, [2654208, 16]);  clone_384 = None
    mm_55: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_549, permute_359);  view_549 = permute_359 = None
    view_550: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_55, [8, 576, 576, 16]);  mm_55 = None
    add_247: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_550, arg519_1);  view_550 = arg519_1 = None
    permute_360: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_247, [0, 3, 1, 2]);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_385: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_360);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_110: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_385, [8, 16, 576, 576]);  clone_385 = None
    clone_386: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
    view_551: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_386, [128, 576, 576]);  clone_386 = None
    expand_111: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_83, [8, 16, 576, 48]);  select_83 = None
    clone_387: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
    view_552: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_387, [128, 576, 48]);  clone_387 = None
    bmm_55: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_551, view_552);  view_551 = view_552 = None
    view_553: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_55, [8, 16, 576, 48]);  bmm_55 = None
    permute_361: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_553, [0, 2, 1, 3]);  view_553 = None
    clone_388: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    view_554: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_388, [8, 576, 768]);  clone_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_555: "f32[4608, 768]" = torch.ops.aten.view.default(view_554, [4608, 768]);  view_554 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(arg520_1, [1, 0]);  arg520_1 = None
    addmm_109: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg521_1, view_555, permute_362);  arg521_1 = view_555 = permute_362 = None
    view_556: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_109, [8, 576, 768]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_389: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_556);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_273: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg55_1, clone_389);  arg55_1 = clone_389 = None
    add_248: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_243, mul_273);  add_243 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_390: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_248, memory_format = torch.contiguous_format)
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_390, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 576, 1]" = var_mean_55[0]
    getitem_111: "f32[8, 576, 1]" = var_mean_55[1];  var_mean_55 = None
    add_249: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_55: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
    sub_83: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_390, getitem_111);  clone_390 = getitem_111 = None
    mul_274: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_55);  sub_83 = rsqrt_55 = None
    mul_275: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_274, arg522_1);  mul_274 = arg522_1 = None
    add_250: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_275, arg523_1);  mul_275 = arg523_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_557: "f32[4608, 768]" = torch.ops.aten.view.default(add_250, [4608, 768]);  add_250 = None
    permute_363: "f32[768, 3072]" = torch.ops.aten.permute.default(arg524_1, [1, 0]);  arg524_1 = None
    addmm_110: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg525_1, view_557, permute_363);  arg525_1 = view_557 = permute_363 = None
    view_558: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_110, [8, 576, 3072]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_276: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_558, 0.5)
    mul_277: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_558, 0.7071067811865476);  view_558 = None
    erf_27: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_277);  mul_277 = None
    add_251: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_278: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_276, add_251);  mul_276 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_391: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_278);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_559: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_391, [4608, 3072]);  clone_391 = None
    permute_364: "f32[3072, 768]" = torch.ops.aten.permute.default(arg526_1, [1, 0]);  arg526_1 = None
    addmm_111: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg527_1, view_559, permute_364);  arg527_1 = view_559 = permute_364 = None
    view_560: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_111, [8, 576, 768]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_392: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_560);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_279: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg56_1, clone_392);  arg56_1 = clone_392 = None
    add_252: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_248, mul_279);  add_248 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_393: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_252, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_393, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 576, 1]" = var_mean_56[0]
    getitem_113: "f32[8, 576, 1]" = var_mean_56[1];  var_mean_56 = None
    add_253: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_56: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_84: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_393, getitem_113);  clone_393 = getitem_113 = None
    mul_280: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_56);  sub_84 = rsqrt_56 = None
    mul_281: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_280, arg528_1);  mul_280 = arg528_1 = None
    add_254: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_281, arg529_1);  mul_281 = arg529_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_561: "f32[4608, 768]" = torch.ops.aten.view.default(add_254, [4608, 768]);  add_254 = None
    permute_365: "f32[768, 2304]" = torch.ops.aten.permute.default(arg530_1, [1, 0]);  arg530_1 = None
    addmm_112: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg531_1, view_561, permute_365);  arg531_1 = view_561 = permute_365 = None
    view_562: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_112, [8, 576, 2304]);  addmm_112 = None
    view_563: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_562, [8, 576, 3, 16, 48]);  view_562 = None
    permute_366: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_563, [2, 0, 3, 1, 4]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_84: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_366, 0, 0)
    mul_282: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_84, 0.14433756729740643);  select_84 = None
    select_85: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_366, 0, 1)
    select_86: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_366, 0, 2);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_367: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_85, [0, 1, 3, 2]);  select_85 = None
    expand_112: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_282, [8, 16, 576, 48]);  mul_282 = None
    clone_394: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
    view_564: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_394, [128, 576, 48]);  clone_394 = None
    expand_113: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_367, [8, 16, 48, 576]);  permute_367 = None
    clone_395: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    view_565: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_395, [128, 48, 576]);  clone_395 = None
    bmm_56: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_564, view_565);  view_564 = view_565 = None
    view_566: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_56, [8, 16, 576, 576]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_368: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_566, [0, 2, 3, 1]);  view_566 = None
    permute_369: "f32[16, 16]" = torch.ops.aten.permute.default(arg532_1, [1, 0]);  arg532_1 = None
    clone_396: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_567: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_396, [2654208, 16]);  clone_396 = None
    mm_56: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_567, permute_369);  view_567 = permute_369 = None
    view_568: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_56, [8, 576, 576, 16]);  mm_56 = None
    add_255: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_568, arg533_1);  view_568 = arg533_1 = None
    permute_370: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_255, [0, 3, 1, 2]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_397: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    amax_28: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_397, [-1], True)
    sub_85: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_397, amax_28);  clone_397 = amax_28 = None
    exp_28: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
    sum_29: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
    div_28: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_371: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_28, [0, 2, 3, 1]);  div_28 = None
    permute_372: "f32[16, 16]" = torch.ops.aten.permute.default(arg534_1, [1, 0]);  arg534_1 = None
    clone_398: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_371, memory_format = torch.contiguous_format);  permute_371 = None
    view_569: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_398, [2654208, 16]);  clone_398 = None
    mm_57: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_569, permute_372);  view_569 = permute_372 = None
    view_570: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_57, [8, 576, 576, 16]);  mm_57 = None
    add_256: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_570, arg535_1);  view_570 = arg535_1 = None
    permute_373: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_256, [0, 3, 1, 2]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_399: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_373);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_114: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_399, [8, 16, 576, 576]);  clone_399 = None
    clone_400: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
    view_571: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_400, [128, 576, 576]);  clone_400 = None
    expand_115: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_86, [8, 16, 576, 48]);  select_86 = None
    clone_401: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
    view_572: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_401, [128, 576, 48]);  clone_401 = None
    bmm_57: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_571, view_572);  view_571 = view_572 = None
    view_573: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_57, [8, 16, 576, 48]);  bmm_57 = None
    permute_374: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    clone_402: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
    view_574: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_402, [8, 576, 768]);  clone_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_575: "f32[4608, 768]" = torch.ops.aten.view.default(view_574, [4608, 768]);  view_574 = None
    permute_375: "f32[768, 768]" = torch.ops.aten.permute.default(arg536_1, [1, 0]);  arg536_1 = None
    addmm_113: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg537_1, view_575, permute_375);  arg537_1 = view_575 = permute_375 = None
    view_576: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_113, [8, 576, 768]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_403: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_576);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_283: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg57_1, clone_403);  arg57_1 = clone_403 = None
    add_257: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_252, mul_283);  add_252 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_404: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format)
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_404, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 576, 1]" = var_mean_57[0]
    getitem_115: "f32[8, 576, 1]" = var_mean_57[1];  var_mean_57 = None
    add_258: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
    rsqrt_57: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_86: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_404, getitem_115);  clone_404 = getitem_115 = None
    mul_284: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_57);  sub_86 = rsqrt_57 = None
    mul_285: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_284, arg538_1);  mul_284 = arg538_1 = None
    add_259: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_285, arg539_1);  mul_285 = arg539_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_577: "f32[4608, 768]" = torch.ops.aten.view.default(add_259, [4608, 768]);  add_259 = None
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(arg540_1, [1, 0]);  arg540_1 = None
    addmm_114: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg541_1, view_577, permute_376);  arg541_1 = view_577 = permute_376 = None
    view_578: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_114, [8, 576, 3072]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_286: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_578, 0.5)
    mul_287: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_578, 0.7071067811865476);  view_578 = None
    erf_28: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
    add_260: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_288: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_286, add_260);  mul_286 = add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_405: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_288);  mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_579: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_405, [4608, 3072]);  clone_405 = None
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(arg542_1, [1, 0]);  arg542_1 = None
    addmm_115: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg543_1, view_579, permute_377);  arg543_1 = view_579 = permute_377 = None
    view_580: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_115, [8, 576, 768]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_406: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_580);  view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_289: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg58_1, clone_406);  arg58_1 = clone_406 = None
    add_261: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_257, mul_289);  add_257 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_407: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_261, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_407, [2], correction = 0, keepdim = True)
    getitem_116: "f32[8, 576, 1]" = var_mean_58[0]
    getitem_117: "f32[8, 576, 1]" = var_mean_58[1];  var_mean_58 = None
    add_262: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
    rsqrt_58: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    sub_87: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_407, getitem_117);  clone_407 = getitem_117 = None
    mul_290: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_58);  sub_87 = rsqrt_58 = None
    mul_291: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_290, arg544_1);  mul_290 = arg544_1 = None
    add_263: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_291, arg545_1);  mul_291 = arg545_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_581: "f32[4608, 768]" = torch.ops.aten.view.default(add_263, [4608, 768]);  add_263 = None
    permute_378: "f32[768, 2304]" = torch.ops.aten.permute.default(arg546_1, [1, 0]);  arg546_1 = None
    addmm_116: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg547_1, view_581, permute_378);  arg547_1 = view_581 = permute_378 = None
    view_582: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_116, [8, 576, 2304]);  addmm_116 = None
    view_583: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_582, [8, 576, 3, 16, 48]);  view_582 = None
    permute_379: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_583, [2, 0, 3, 1, 4]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_87: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_379, 0, 0)
    mul_292: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_87, 0.14433756729740643);  select_87 = None
    select_88: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_379, 0, 1)
    select_89: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_379, 0, 2);  permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_380: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_88, [0, 1, 3, 2]);  select_88 = None
    expand_116: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_292, [8, 16, 576, 48]);  mul_292 = None
    clone_408: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
    view_584: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_408, [128, 576, 48]);  clone_408 = None
    expand_117: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_380, [8, 16, 48, 576]);  permute_380 = None
    clone_409: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
    view_585: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_409, [128, 48, 576]);  clone_409 = None
    bmm_58: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_584, view_585);  view_584 = view_585 = None
    view_586: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_58, [8, 16, 576, 576]);  bmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_381: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_586, [0, 2, 3, 1]);  view_586 = None
    permute_382: "f32[16, 16]" = torch.ops.aten.permute.default(arg548_1, [1, 0]);  arg548_1 = None
    clone_410: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_381, memory_format = torch.contiguous_format);  permute_381 = None
    view_587: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_410, [2654208, 16]);  clone_410 = None
    mm_58: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_587, permute_382);  view_587 = permute_382 = None
    view_588: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_58, [8, 576, 576, 16]);  mm_58 = None
    add_264: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_588, arg549_1);  view_588 = arg549_1 = None
    permute_383: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_264, [0, 3, 1, 2]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_411: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
    amax_29: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_411, [-1], True)
    sub_88: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_411, amax_29);  clone_411 = amax_29 = None
    exp_29: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_88);  sub_88 = None
    sum_30: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
    div_29: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_384: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_29, [0, 2, 3, 1]);  div_29 = None
    permute_385: "f32[16, 16]" = torch.ops.aten.permute.default(arg550_1, [1, 0]);  arg550_1 = None
    clone_412: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    view_589: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_412, [2654208, 16]);  clone_412 = None
    mm_59: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_589, permute_385);  view_589 = permute_385 = None
    view_590: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_59, [8, 576, 576, 16]);  mm_59 = None
    add_265: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_590, arg551_1);  view_590 = arg551_1 = None
    permute_386: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_265, [0, 3, 1, 2]);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_413: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_386);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_118: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_413, [8, 16, 576, 576]);  clone_413 = None
    clone_414: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_118, memory_format = torch.contiguous_format);  expand_118 = None
    view_591: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_414, [128, 576, 576]);  clone_414 = None
    expand_119: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_89, [8, 16, 576, 48]);  select_89 = None
    clone_415: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
    view_592: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_415, [128, 576, 48]);  clone_415 = None
    bmm_59: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_591, view_592);  view_591 = view_592 = None
    view_593: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_59, [8, 16, 576, 48]);  bmm_59 = None
    permute_387: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    clone_416: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_594: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_416, [8, 576, 768]);  clone_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_595: "f32[4608, 768]" = torch.ops.aten.view.default(view_594, [4608, 768]);  view_594 = None
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(arg552_1, [1, 0]);  arg552_1 = None
    addmm_117: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg553_1, view_595, permute_388);  arg553_1 = view_595 = permute_388 = None
    view_596: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_117, [8, 576, 768]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_417: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_596);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_293: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg59_1, clone_417);  arg59_1 = clone_417 = None
    add_266: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_261, mul_293);  add_261 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_418: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_266, memory_format = torch.contiguous_format)
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_418, [2], correction = 0, keepdim = True)
    getitem_118: "f32[8, 576, 1]" = var_mean_59[0]
    getitem_119: "f32[8, 576, 1]" = var_mean_59[1];  var_mean_59 = None
    add_267: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
    rsqrt_59: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_89: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_418, getitem_119);  clone_418 = getitem_119 = None
    mul_294: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_59);  sub_89 = rsqrt_59 = None
    mul_295: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_294, arg554_1);  mul_294 = arg554_1 = None
    add_268: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_295, arg555_1);  mul_295 = arg555_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_597: "f32[4608, 768]" = torch.ops.aten.view.default(add_268, [4608, 768]);  add_268 = None
    permute_389: "f32[768, 3072]" = torch.ops.aten.permute.default(arg556_1, [1, 0]);  arg556_1 = None
    addmm_118: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg557_1, view_597, permute_389);  arg557_1 = view_597 = permute_389 = None
    view_598: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_118, [8, 576, 3072]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_296: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_598, 0.5)
    mul_297: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_598, 0.7071067811865476);  view_598 = None
    erf_29: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_297);  mul_297 = None
    add_269: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_298: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_296, add_269);  mul_296 = add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_419: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_298);  mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_599: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_419, [4608, 3072]);  clone_419 = None
    permute_390: "f32[3072, 768]" = torch.ops.aten.permute.default(arg558_1, [1, 0]);  arg558_1 = None
    addmm_119: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg559_1, view_599, permute_390);  arg559_1 = view_599 = permute_390 = None
    view_600: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_119, [8, 576, 768]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_420: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_600);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_299: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg60_1, clone_420);  arg60_1 = clone_420 = None
    add_270: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_266, mul_299);  add_266 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_421: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_270, memory_format = torch.contiguous_format)
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_421, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 576, 1]" = var_mean_60[0]
    getitem_121: "f32[8, 576, 1]" = var_mean_60[1];  var_mean_60 = None
    add_271: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_60: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
    sub_90: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_421, getitem_121);  clone_421 = getitem_121 = None
    mul_300: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_60);  sub_90 = rsqrt_60 = None
    mul_301: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_300, arg560_1);  mul_300 = arg560_1 = None
    add_272: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_301, arg561_1);  mul_301 = arg561_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_601: "f32[4608, 768]" = torch.ops.aten.view.default(add_272, [4608, 768]);  add_272 = None
    permute_391: "f32[768, 2304]" = torch.ops.aten.permute.default(arg562_1, [1, 0]);  arg562_1 = None
    addmm_120: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg563_1, view_601, permute_391);  arg563_1 = view_601 = permute_391 = None
    view_602: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_120, [8, 576, 2304]);  addmm_120 = None
    view_603: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_602, [8, 576, 3, 16, 48]);  view_602 = None
    permute_392: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_603, [2, 0, 3, 1, 4]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_90: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_392, 0, 0)
    mul_302: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_90, 0.14433756729740643);  select_90 = None
    select_91: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_392, 0, 1)
    select_92: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_392, 0, 2);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_393: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_91, [0, 1, 3, 2]);  select_91 = None
    expand_120: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_302, [8, 16, 576, 48]);  mul_302 = None
    clone_422: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
    view_604: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_422, [128, 576, 48]);  clone_422 = None
    expand_121: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_393, [8, 16, 48, 576]);  permute_393 = None
    clone_423: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
    view_605: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_423, [128, 48, 576]);  clone_423 = None
    bmm_60: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_604, view_605);  view_604 = view_605 = None
    view_606: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_60, [8, 16, 576, 576]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_394: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_606, [0, 2, 3, 1]);  view_606 = None
    permute_395: "f32[16, 16]" = torch.ops.aten.permute.default(arg564_1, [1, 0]);  arg564_1 = None
    clone_424: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_394, memory_format = torch.contiguous_format);  permute_394 = None
    view_607: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_424, [2654208, 16]);  clone_424 = None
    mm_60: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_607, permute_395);  view_607 = permute_395 = None
    view_608: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_60, [8, 576, 576, 16]);  mm_60 = None
    add_273: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_608, arg565_1);  view_608 = arg565_1 = None
    permute_396: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_273, [0, 3, 1, 2]);  add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_425: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_396, memory_format = torch.contiguous_format);  permute_396 = None
    amax_30: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_425, [-1], True)
    sub_91: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_425, amax_30);  clone_425 = amax_30 = None
    exp_30: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_91);  sub_91 = None
    sum_31: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
    div_30: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_397: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_30, [0, 2, 3, 1]);  div_30 = None
    permute_398: "f32[16, 16]" = torch.ops.aten.permute.default(arg566_1, [1, 0]);  arg566_1 = None
    clone_426: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_397, memory_format = torch.contiguous_format);  permute_397 = None
    view_609: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_426, [2654208, 16]);  clone_426 = None
    mm_61: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_609, permute_398);  view_609 = permute_398 = None
    view_610: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_61, [8, 576, 576, 16]);  mm_61 = None
    add_274: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_610, arg567_1);  view_610 = arg567_1 = None
    permute_399: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_274, [0, 3, 1, 2]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_427: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_399);  permute_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_122: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_427, [8, 16, 576, 576]);  clone_427 = None
    clone_428: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
    view_611: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_428, [128, 576, 576]);  clone_428 = None
    expand_123: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_92, [8, 16, 576, 48]);  select_92 = None
    clone_429: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
    view_612: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_429, [128, 576, 48]);  clone_429 = None
    bmm_61: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_611, view_612);  view_611 = view_612 = None
    view_613: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_61, [8, 16, 576, 48]);  bmm_61 = None
    permute_400: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
    clone_430: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
    view_614: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_430, [8, 576, 768]);  clone_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_615: "f32[4608, 768]" = torch.ops.aten.view.default(view_614, [4608, 768]);  view_614 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(arg568_1, [1, 0]);  arg568_1 = None
    addmm_121: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg569_1, view_615, permute_401);  arg569_1 = view_615 = permute_401 = None
    view_616: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_121, [8, 576, 768]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_431: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_616);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_303: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg61_1, clone_431);  arg61_1 = clone_431 = None
    add_275: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_270, mul_303);  add_270 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_432: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_275, memory_format = torch.contiguous_format)
    var_mean_61 = torch.ops.aten.var_mean.correction(clone_432, [2], correction = 0, keepdim = True)
    getitem_122: "f32[8, 576, 1]" = var_mean_61[0]
    getitem_123: "f32[8, 576, 1]" = var_mean_61[1];  var_mean_61 = None
    add_276: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
    rsqrt_61: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    sub_92: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_432, getitem_123);  clone_432 = getitem_123 = None
    mul_304: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_61);  sub_92 = rsqrt_61 = None
    mul_305: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_304, arg570_1);  mul_304 = arg570_1 = None
    add_277: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_305, arg571_1);  mul_305 = arg571_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_617: "f32[4608, 768]" = torch.ops.aten.view.default(add_277, [4608, 768]);  add_277 = None
    permute_402: "f32[768, 3072]" = torch.ops.aten.permute.default(arg572_1, [1, 0]);  arg572_1 = None
    addmm_122: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg573_1, view_617, permute_402);  arg573_1 = view_617 = permute_402 = None
    view_618: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_122, [8, 576, 3072]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_306: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_618, 0.5)
    mul_307: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_618, 0.7071067811865476);  view_618 = None
    erf_30: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_307);  mul_307 = None
    add_278: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_308: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_306, add_278);  mul_306 = add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_433: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_308);  mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_619: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_433, [4608, 3072]);  clone_433 = None
    permute_403: "f32[3072, 768]" = torch.ops.aten.permute.default(arg574_1, [1, 0]);  arg574_1 = None
    addmm_123: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg575_1, view_619, permute_403);  arg575_1 = view_619 = permute_403 = None
    view_620: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_123, [8, 576, 768]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_434: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_620);  view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_309: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg62_1, clone_434);  arg62_1 = clone_434 = None
    add_279: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_275, mul_309);  add_275 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_435: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format)
    var_mean_62 = torch.ops.aten.var_mean.correction(clone_435, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 576, 1]" = var_mean_62[0]
    getitem_125: "f32[8, 576, 1]" = var_mean_62[1];  var_mean_62 = None
    add_280: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_62: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
    sub_93: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_435, getitem_125);  clone_435 = getitem_125 = None
    mul_310: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_62);  sub_93 = rsqrt_62 = None
    mul_311: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_310, arg576_1);  mul_310 = arg576_1 = None
    add_281: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_311, arg577_1);  mul_311 = arg577_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_621: "f32[4608, 768]" = torch.ops.aten.view.default(add_281, [4608, 768]);  add_281 = None
    permute_404: "f32[768, 2304]" = torch.ops.aten.permute.default(arg578_1, [1, 0]);  arg578_1 = None
    addmm_124: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg579_1, view_621, permute_404);  arg579_1 = view_621 = permute_404 = None
    view_622: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_124, [8, 576, 2304]);  addmm_124 = None
    view_623: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_622, [8, 576, 3, 16, 48]);  view_622 = None
    permute_405: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_623, [2, 0, 3, 1, 4]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_93: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_405, 0, 0)
    mul_312: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_93, 0.14433756729740643);  select_93 = None
    select_94: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_405, 0, 1)
    select_95: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_405, 0, 2);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_406: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_94, [0, 1, 3, 2]);  select_94 = None
    expand_124: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_312, [8, 16, 576, 48]);  mul_312 = None
    clone_436: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
    view_624: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_436, [128, 576, 48]);  clone_436 = None
    expand_125: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_406, [8, 16, 48, 576]);  permute_406 = None
    clone_437: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
    view_625: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_437, [128, 48, 576]);  clone_437 = None
    bmm_62: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_624, view_625);  view_624 = view_625 = None
    view_626: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_62, [8, 16, 576, 576]);  bmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_407: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_626, [0, 2, 3, 1]);  view_626 = None
    permute_408: "f32[16, 16]" = torch.ops.aten.permute.default(arg580_1, [1, 0]);  arg580_1 = None
    clone_438: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    view_627: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_438, [2654208, 16]);  clone_438 = None
    mm_62: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_627, permute_408);  view_627 = permute_408 = None
    view_628: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_62, [8, 576, 576, 16]);  mm_62 = None
    add_282: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_628, arg581_1);  view_628 = arg581_1 = None
    permute_409: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_282, [0, 3, 1, 2]);  add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_439: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    amax_31: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_439, [-1], True)
    sub_94: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_439, amax_31);  clone_439 = amax_31 = None
    exp_31: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_94);  sub_94 = None
    sum_32: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
    div_31: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_410: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_31, [0, 2, 3, 1]);  div_31 = None
    permute_411: "f32[16, 16]" = torch.ops.aten.permute.default(arg582_1, [1, 0]);  arg582_1 = None
    clone_440: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    view_629: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_440, [2654208, 16]);  clone_440 = None
    mm_63: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_629, permute_411);  view_629 = permute_411 = None
    view_630: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_63, [8, 576, 576, 16]);  mm_63 = None
    add_283: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_630, arg583_1);  view_630 = arg583_1 = None
    permute_412: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_283, [0, 3, 1, 2]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_441: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_412);  permute_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_126: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_441, [8, 16, 576, 576]);  clone_441 = None
    clone_442: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_126, memory_format = torch.contiguous_format);  expand_126 = None
    view_631: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_442, [128, 576, 576]);  clone_442 = None
    expand_127: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_95, [8, 16, 576, 48]);  select_95 = None
    clone_443: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
    view_632: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_443, [128, 576, 48]);  clone_443 = None
    bmm_63: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_631, view_632);  view_631 = view_632 = None
    view_633: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_63, [8, 16, 576, 48]);  bmm_63 = None
    permute_413: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_633, [0, 2, 1, 3]);  view_633 = None
    clone_444: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
    view_634: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_444, [8, 576, 768]);  clone_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_635: "f32[4608, 768]" = torch.ops.aten.view.default(view_634, [4608, 768]);  view_634 = None
    permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(arg584_1, [1, 0]);  arg584_1 = None
    addmm_125: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg585_1, view_635, permute_414);  arg585_1 = view_635 = permute_414 = None
    view_636: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_125, [8, 576, 768]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_445: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_636);  view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_313: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg63_1, clone_445);  arg63_1 = clone_445 = None
    add_284: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_279, mul_313);  add_279 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_446: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_284, memory_format = torch.contiguous_format)
    var_mean_63 = torch.ops.aten.var_mean.correction(clone_446, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 576, 1]" = var_mean_63[0]
    getitem_127: "f32[8, 576, 1]" = var_mean_63[1];  var_mean_63 = None
    add_285: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_63: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    sub_95: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_446, getitem_127);  clone_446 = getitem_127 = None
    mul_314: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_63);  sub_95 = rsqrt_63 = None
    mul_315: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_314, arg586_1);  mul_314 = arg586_1 = None
    add_286: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_315, arg587_1);  mul_315 = arg587_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_637: "f32[4608, 768]" = torch.ops.aten.view.default(add_286, [4608, 768]);  add_286 = None
    permute_415: "f32[768, 3072]" = torch.ops.aten.permute.default(arg588_1, [1, 0]);  arg588_1 = None
    addmm_126: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg589_1, view_637, permute_415);  arg589_1 = view_637 = permute_415 = None
    view_638: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_126, [8, 576, 3072]);  addmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_316: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_638, 0.5)
    mul_317: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_638, 0.7071067811865476);  view_638 = None
    erf_31: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_317);  mul_317 = None
    add_287: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_318: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_316, add_287);  mul_316 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_447: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_318);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_639: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_447, [4608, 3072]);  clone_447 = None
    permute_416: "f32[3072, 768]" = torch.ops.aten.permute.default(arg590_1, [1, 0]);  arg590_1 = None
    addmm_127: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg591_1, view_639, permute_416);  arg591_1 = view_639 = permute_416 = None
    view_640: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_127, [8, 576, 768]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_448: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_640);  view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_319: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg64_1, clone_448);  arg64_1 = clone_448 = None
    add_288: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_284, mul_319);  add_284 = mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_449: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
    var_mean_64 = torch.ops.aten.var_mean.correction(clone_449, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 576, 1]" = var_mean_64[0]
    getitem_129: "f32[8, 576, 1]" = var_mean_64[1];  var_mean_64 = None
    add_289: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
    rsqrt_64: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_96: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_449, getitem_129);  clone_449 = getitem_129 = None
    mul_320: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_64);  sub_96 = rsqrt_64 = None
    mul_321: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_320, arg592_1);  mul_320 = arg592_1 = None
    add_290: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_321, arg593_1);  mul_321 = arg593_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_641: "f32[4608, 768]" = torch.ops.aten.view.default(add_290, [4608, 768]);  add_290 = None
    permute_417: "f32[768, 2304]" = torch.ops.aten.permute.default(arg594_1, [1, 0]);  arg594_1 = None
    addmm_128: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg595_1, view_641, permute_417);  arg595_1 = view_641 = permute_417 = None
    view_642: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_128, [8, 576, 2304]);  addmm_128 = None
    view_643: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_642, [8, 576, 3, 16, 48]);  view_642 = None
    permute_418: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_643, [2, 0, 3, 1, 4]);  view_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_96: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_418, 0, 0)
    mul_322: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_96, 0.14433756729740643);  select_96 = None
    select_97: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_418, 0, 1)
    select_98: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_418, 0, 2);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_419: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_97, [0, 1, 3, 2]);  select_97 = None
    expand_128: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_322, [8, 16, 576, 48]);  mul_322 = None
    clone_450: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    view_644: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_450, [128, 576, 48]);  clone_450 = None
    expand_129: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_419, [8, 16, 48, 576]);  permute_419 = None
    clone_451: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    view_645: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_451, [128, 48, 576]);  clone_451 = None
    bmm_64: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_644, view_645);  view_644 = view_645 = None
    view_646: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_64, [8, 16, 576, 576]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_420: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_646, [0, 2, 3, 1]);  view_646 = None
    permute_421: "f32[16, 16]" = torch.ops.aten.permute.default(arg596_1, [1, 0]);  arg596_1 = None
    clone_452: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_647: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_452, [2654208, 16]);  clone_452 = None
    mm_64: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_647, permute_421);  view_647 = permute_421 = None
    view_648: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_64, [8, 576, 576, 16]);  mm_64 = None
    add_291: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_648, arg597_1);  view_648 = arg597_1 = None
    permute_422: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_291, [0, 3, 1, 2]);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_453: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_422, memory_format = torch.contiguous_format);  permute_422 = None
    amax_32: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_453, [-1], True)
    sub_97: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_453, amax_32);  clone_453 = amax_32 = None
    exp_32: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_97);  sub_97 = None
    sum_33: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
    div_32: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_423: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_32, [0, 2, 3, 1]);  div_32 = None
    permute_424: "f32[16, 16]" = torch.ops.aten.permute.default(arg598_1, [1, 0]);  arg598_1 = None
    clone_454: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
    view_649: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_454, [2654208, 16]);  clone_454 = None
    mm_65: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_649, permute_424);  view_649 = permute_424 = None
    view_650: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_65, [8, 576, 576, 16]);  mm_65 = None
    add_292: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_650, arg599_1);  view_650 = arg599_1 = None
    permute_425: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_292, [0, 3, 1, 2]);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_455: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_425);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_130: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_455, [8, 16, 576, 576]);  clone_455 = None
    clone_456: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
    view_651: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_456, [128, 576, 576]);  clone_456 = None
    expand_131: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_98, [8, 16, 576, 48]);  select_98 = None
    clone_457: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
    view_652: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_457, [128, 576, 48]);  clone_457 = None
    bmm_65: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_651, view_652);  view_651 = view_652 = None
    view_653: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_65, [8, 16, 576, 48]);  bmm_65 = None
    permute_426: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    clone_458: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_654: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_458, [8, 576, 768]);  clone_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_655: "f32[4608, 768]" = torch.ops.aten.view.default(view_654, [4608, 768]);  view_654 = None
    permute_427: "f32[768, 768]" = torch.ops.aten.permute.default(arg600_1, [1, 0]);  arg600_1 = None
    addmm_129: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg601_1, view_655, permute_427);  arg601_1 = view_655 = permute_427 = None
    view_656: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_129, [8, 576, 768]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_459: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_656);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_323: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg65_1, clone_459);  arg65_1 = clone_459 = None
    add_293: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_288, mul_323);  add_288 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_460: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_293, memory_format = torch.contiguous_format)
    var_mean_65 = torch.ops.aten.var_mean.correction(clone_460, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 576, 1]" = var_mean_65[0]
    getitem_131: "f32[8, 576, 1]" = var_mean_65[1];  var_mean_65 = None
    add_294: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_65: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    sub_98: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_460, getitem_131);  clone_460 = getitem_131 = None
    mul_324: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_65);  sub_98 = rsqrt_65 = None
    mul_325: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_324, arg602_1);  mul_324 = arg602_1 = None
    add_295: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_325, arg603_1);  mul_325 = arg603_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_657: "f32[4608, 768]" = torch.ops.aten.view.default(add_295, [4608, 768]);  add_295 = None
    permute_428: "f32[768, 3072]" = torch.ops.aten.permute.default(arg604_1, [1, 0]);  arg604_1 = None
    addmm_130: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg605_1, view_657, permute_428);  arg605_1 = view_657 = permute_428 = None
    view_658: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_130, [8, 576, 3072]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_326: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_658, 0.5)
    mul_327: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_658, 0.7071067811865476);  view_658 = None
    erf_32: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_327);  mul_327 = None
    add_296: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_328: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_326, add_296);  mul_326 = add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_461: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_328);  mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_659: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_461, [4608, 3072]);  clone_461 = None
    permute_429: "f32[3072, 768]" = torch.ops.aten.permute.default(arg606_1, [1, 0]);  arg606_1 = None
    addmm_131: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg607_1, view_659, permute_429);  arg607_1 = view_659 = permute_429 = None
    view_660: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_131, [8, 576, 768]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_462: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_660);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_329: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg66_1, clone_462);  arg66_1 = clone_462 = None
    add_297: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_293, mul_329);  add_293 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_463: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_297, memory_format = torch.contiguous_format)
    var_mean_66 = torch.ops.aten.var_mean.correction(clone_463, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 576, 1]" = var_mean_66[0]
    getitem_133: "f32[8, 576, 1]" = var_mean_66[1];  var_mean_66 = None
    add_298: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_66: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
    sub_99: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_463, getitem_133);  clone_463 = getitem_133 = None
    mul_330: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_66);  sub_99 = rsqrt_66 = None
    mul_331: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_330, arg608_1);  mul_330 = arg608_1 = None
    add_299: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_331, arg609_1);  mul_331 = arg609_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_661: "f32[4608, 768]" = torch.ops.aten.view.default(add_299, [4608, 768]);  add_299 = None
    permute_430: "f32[768, 2304]" = torch.ops.aten.permute.default(arg610_1, [1, 0]);  arg610_1 = None
    addmm_132: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg611_1, view_661, permute_430);  arg611_1 = view_661 = permute_430 = None
    view_662: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_132, [8, 576, 2304]);  addmm_132 = None
    view_663: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_662, [8, 576, 3, 16, 48]);  view_662 = None
    permute_431: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_663, [2, 0, 3, 1, 4]);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_99: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_431, 0, 0)
    mul_332: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_99, 0.14433756729740643);  select_99 = None
    select_100: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_431, 0, 1)
    select_101: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_431, 0, 2);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_432: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_100, [0, 1, 3, 2]);  select_100 = None
    expand_132: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_332, [8, 16, 576, 48]);  mul_332 = None
    clone_464: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
    view_664: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_464, [128, 576, 48]);  clone_464 = None
    expand_133: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_432, [8, 16, 48, 576]);  permute_432 = None
    clone_465: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
    view_665: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_465, [128, 48, 576]);  clone_465 = None
    bmm_66: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_664, view_665);  view_664 = view_665 = None
    view_666: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_66, [8, 16, 576, 576]);  bmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_433: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_666, [0, 2, 3, 1]);  view_666 = None
    permute_434: "f32[16, 16]" = torch.ops.aten.permute.default(arg612_1, [1, 0]);  arg612_1 = None
    clone_466: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_667: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_466, [2654208, 16]);  clone_466 = None
    mm_66: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_667, permute_434);  view_667 = permute_434 = None
    view_668: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_66, [8, 576, 576, 16]);  mm_66 = None
    add_300: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_668, arg613_1);  view_668 = arg613_1 = None
    permute_435: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_300, [0, 3, 1, 2]);  add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_467: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
    amax_33: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_467, [-1], True)
    sub_100: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_467, amax_33);  clone_467 = amax_33 = None
    exp_33: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_100);  sub_100 = None
    sum_34: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
    div_33: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_436: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_33, [0, 2, 3, 1]);  div_33 = None
    permute_437: "f32[16, 16]" = torch.ops.aten.permute.default(arg614_1, [1, 0]);  arg614_1 = None
    clone_468: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
    view_669: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_468, [2654208, 16]);  clone_468 = None
    mm_67: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_669, permute_437);  view_669 = permute_437 = None
    view_670: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_67, [8, 576, 576, 16]);  mm_67 = None
    add_301: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_670, arg615_1);  view_670 = arg615_1 = None
    permute_438: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_301, [0, 3, 1, 2]);  add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_469: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_438);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_134: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_469, [8, 16, 576, 576]);  clone_469 = None
    clone_470: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
    view_671: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_470, [128, 576, 576]);  clone_470 = None
    expand_135: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_101, [8, 16, 576, 48]);  select_101 = None
    clone_471: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
    view_672: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_471, [128, 576, 48]);  clone_471 = None
    bmm_67: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_671, view_672);  view_671 = view_672 = None
    view_673: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_67, [8, 16, 576, 48]);  bmm_67 = None
    permute_439: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
    clone_472: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_439, memory_format = torch.contiguous_format);  permute_439 = None
    view_674: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_472, [8, 576, 768]);  clone_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_675: "f32[4608, 768]" = torch.ops.aten.view.default(view_674, [4608, 768]);  view_674 = None
    permute_440: "f32[768, 768]" = torch.ops.aten.permute.default(arg616_1, [1, 0]);  arg616_1 = None
    addmm_133: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg617_1, view_675, permute_440);  arg617_1 = view_675 = permute_440 = None
    view_676: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_133, [8, 576, 768]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_473: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_676);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_333: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg67_1, clone_473);  arg67_1 = clone_473 = None
    add_302: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_297, mul_333);  add_297 = mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_474: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
    var_mean_67 = torch.ops.aten.var_mean.correction(clone_474, [2], correction = 0, keepdim = True)
    getitem_134: "f32[8, 576, 1]" = var_mean_67[0]
    getitem_135: "f32[8, 576, 1]" = var_mean_67[1];  var_mean_67 = None
    add_303: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
    rsqrt_67: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
    sub_101: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_474, getitem_135);  clone_474 = getitem_135 = None
    mul_334: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_67);  sub_101 = rsqrt_67 = None
    mul_335: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_334, arg618_1);  mul_334 = arg618_1 = None
    add_304: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_335, arg619_1);  mul_335 = arg619_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_677: "f32[4608, 768]" = torch.ops.aten.view.default(add_304, [4608, 768]);  add_304 = None
    permute_441: "f32[768, 3072]" = torch.ops.aten.permute.default(arg620_1, [1, 0]);  arg620_1 = None
    addmm_134: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg621_1, view_677, permute_441);  arg621_1 = view_677 = permute_441 = None
    view_678: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_134, [8, 576, 3072]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_336: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_678, 0.5)
    mul_337: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_678, 0.7071067811865476);  view_678 = None
    erf_33: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_337);  mul_337 = None
    add_305: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_338: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_336, add_305);  mul_336 = add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_475: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_338);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_679: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_475, [4608, 3072]);  clone_475 = None
    permute_442: "f32[3072, 768]" = torch.ops.aten.permute.default(arg622_1, [1, 0]);  arg622_1 = None
    addmm_135: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg623_1, view_679, permute_442);  arg623_1 = view_679 = permute_442 = None
    view_680: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_135, [8, 576, 768]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_476: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_680);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_339: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg68_1, clone_476);  arg68_1 = clone_476 = None
    add_306: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_302, mul_339);  add_302 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_477: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_306, memory_format = torch.contiguous_format)
    var_mean_68 = torch.ops.aten.var_mean.correction(clone_477, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 576, 1]" = var_mean_68[0]
    getitem_137: "f32[8, 576, 1]" = var_mean_68[1];  var_mean_68 = None
    add_307: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
    rsqrt_68: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
    sub_102: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_477, getitem_137);  clone_477 = getitem_137 = None
    mul_340: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_68);  sub_102 = rsqrt_68 = None
    mul_341: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_340, arg624_1);  mul_340 = arg624_1 = None
    add_308: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_341, arg625_1);  mul_341 = arg625_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_681: "f32[4608, 768]" = torch.ops.aten.view.default(add_308, [4608, 768]);  add_308 = None
    permute_443: "f32[768, 2304]" = torch.ops.aten.permute.default(arg626_1, [1, 0]);  arg626_1 = None
    addmm_136: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg627_1, view_681, permute_443);  arg627_1 = view_681 = permute_443 = None
    view_682: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_136, [8, 576, 2304]);  addmm_136 = None
    view_683: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_682, [8, 576, 3, 16, 48]);  view_682 = None
    permute_444: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_683, [2, 0, 3, 1, 4]);  view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_102: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_444, 0, 0)
    mul_342: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_102, 0.14433756729740643);  select_102 = None
    select_103: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_444, 0, 1)
    select_104: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_444, 0, 2);  permute_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_445: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_103, [0, 1, 3, 2]);  select_103 = None
    expand_136: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_342, [8, 16, 576, 48]);  mul_342 = None
    clone_478: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
    view_684: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_478, [128, 576, 48]);  clone_478 = None
    expand_137: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_445, [8, 16, 48, 576]);  permute_445 = None
    clone_479: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    view_685: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_479, [128, 48, 576]);  clone_479 = None
    bmm_68: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_684, view_685);  view_684 = view_685 = None
    view_686: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_68, [8, 16, 576, 576]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_446: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_686, [0, 2, 3, 1]);  view_686 = None
    permute_447: "f32[16, 16]" = torch.ops.aten.permute.default(arg628_1, [1, 0]);  arg628_1 = None
    clone_480: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
    view_687: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_480, [2654208, 16]);  clone_480 = None
    mm_68: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_687, permute_447);  view_687 = permute_447 = None
    view_688: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_68, [8, 576, 576, 16]);  mm_68 = None
    add_309: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_688, arg629_1);  view_688 = arg629_1 = None
    permute_448: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_309, [0, 3, 1, 2]);  add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_481: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    amax_34: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_481, [-1], True)
    sub_103: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_481, amax_34);  clone_481 = amax_34 = None
    exp_34: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_103);  sub_103 = None
    sum_35: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
    div_34: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_449: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_34, [0, 2, 3, 1]);  div_34 = None
    permute_450: "f32[16, 16]" = torch.ops.aten.permute.default(arg630_1, [1, 0]);  arg630_1 = None
    clone_482: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_449, memory_format = torch.contiguous_format);  permute_449 = None
    view_689: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_482, [2654208, 16]);  clone_482 = None
    mm_69: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_689, permute_450);  view_689 = permute_450 = None
    view_690: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_69, [8, 576, 576, 16]);  mm_69 = None
    add_310: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_690, arg631_1);  view_690 = arg631_1 = None
    permute_451: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_310, [0, 3, 1, 2]);  add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_483: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_451);  permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_138: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_483, [8, 16, 576, 576]);  clone_483 = None
    clone_484: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
    view_691: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_484, [128, 576, 576]);  clone_484 = None
    expand_139: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_104, [8, 16, 576, 48]);  select_104 = None
    clone_485: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
    view_692: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_485, [128, 576, 48]);  clone_485 = None
    bmm_69: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_691, view_692);  view_691 = view_692 = None
    view_693: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_69, [8, 16, 576, 48]);  bmm_69 = None
    permute_452: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
    clone_486: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_694: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_486, [8, 576, 768]);  clone_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_695: "f32[4608, 768]" = torch.ops.aten.view.default(view_694, [4608, 768]);  view_694 = None
    permute_453: "f32[768, 768]" = torch.ops.aten.permute.default(arg632_1, [1, 0]);  arg632_1 = None
    addmm_137: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg633_1, view_695, permute_453);  arg633_1 = view_695 = permute_453 = None
    view_696: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_137, [8, 576, 768]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_487: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_696);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_343: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg69_1, clone_487);  arg69_1 = clone_487 = None
    add_311: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_306, mul_343);  add_306 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_488: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_311, memory_format = torch.contiguous_format)
    var_mean_69 = torch.ops.aten.var_mean.correction(clone_488, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 576, 1]" = var_mean_69[0]
    getitem_139: "f32[8, 576, 1]" = var_mean_69[1];  var_mean_69 = None
    add_312: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_69: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_312);  add_312 = None
    sub_104: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_488, getitem_139);  clone_488 = getitem_139 = None
    mul_344: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_69);  sub_104 = rsqrt_69 = None
    mul_345: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_344, arg634_1);  mul_344 = arg634_1 = None
    add_313: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_345, arg635_1);  mul_345 = arg635_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_697: "f32[4608, 768]" = torch.ops.aten.view.default(add_313, [4608, 768]);  add_313 = None
    permute_454: "f32[768, 3072]" = torch.ops.aten.permute.default(arg636_1, [1, 0]);  arg636_1 = None
    addmm_138: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg637_1, view_697, permute_454);  arg637_1 = view_697 = permute_454 = None
    view_698: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_138, [8, 576, 3072]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_346: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_698, 0.5)
    mul_347: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_698, 0.7071067811865476);  view_698 = None
    erf_34: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_347);  mul_347 = None
    add_314: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_348: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_346, add_314);  mul_346 = add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_489: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_348);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_699: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_489, [4608, 3072]);  clone_489 = None
    permute_455: "f32[3072, 768]" = torch.ops.aten.permute.default(arg638_1, [1, 0]);  arg638_1 = None
    addmm_139: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg639_1, view_699, permute_455);  arg639_1 = view_699 = permute_455 = None
    view_700: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_139, [8, 576, 768]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_490: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_700);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_349: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg70_1, clone_490);  arg70_1 = clone_490 = None
    add_315: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_311, mul_349);  add_311 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_491: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_315, memory_format = torch.contiguous_format)
    var_mean_70 = torch.ops.aten.var_mean.correction(clone_491, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 576, 1]" = var_mean_70[0]
    getitem_141: "f32[8, 576, 1]" = var_mean_70[1];  var_mean_70 = None
    add_316: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_70: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
    sub_105: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_491, getitem_141);  clone_491 = getitem_141 = None
    mul_350: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_70);  sub_105 = rsqrt_70 = None
    mul_351: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_350, arg640_1);  mul_350 = arg640_1 = None
    add_317: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_351, arg641_1);  mul_351 = arg641_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_701: "f32[4608, 768]" = torch.ops.aten.view.default(add_317, [4608, 768]);  add_317 = None
    permute_456: "f32[768, 2304]" = torch.ops.aten.permute.default(arg642_1, [1, 0]);  arg642_1 = None
    addmm_140: "f32[4608, 2304]" = torch.ops.aten.addmm.default(arg643_1, view_701, permute_456);  arg643_1 = view_701 = permute_456 = None
    view_702: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_140, [8, 576, 2304]);  addmm_140 = None
    view_703: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_702, [8, 576, 3, 16, 48]);  view_702 = None
    permute_457: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_703, [2, 0, 3, 1, 4]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_105: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_457, 0, 0)
    mul_352: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_105, 0.14433756729740643);  select_105 = None
    select_106: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_457, 0, 1)
    select_107: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_457, 0, 2);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_458: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_106, [0, 1, 3, 2]);  select_106 = None
    expand_140: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_352, [8, 16, 576, 48]);  mul_352 = None
    clone_492: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
    view_704: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_492, [128, 576, 48]);  clone_492 = None
    expand_141: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_458, [8, 16, 48, 576]);  permute_458 = None
    clone_493: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
    view_705: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_493, [128, 48, 576]);  clone_493 = None
    bmm_70: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_704, view_705);  view_704 = view_705 = None
    view_706: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_70, [8, 16, 576, 576]);  bmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_459: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_706, [0, 2, 3, 1]);  view_706 = None
    permute_460: "f32[16, 16]" = torch.ops.aten.permute.default(arg644_1, [1, 0]);  arg644_1 = None
    clone_494: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_707: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_494, [2654208, 16]);  clone_494 = None
    mm_70: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_707, permute_460);  view_707 = permute_460 = None
    view_708: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_70, [8, 576, 576, 16]);  mm_70 = None
    add_318: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_708, arg645_1);  view_708 = arg645_1 = None
    permute_461: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_318, [0, 3, 1, 2]);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_495: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
    amax_35: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_495, [-1], True)
    sub_106: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_495, amax_35);  clone_495 = amax_35 = None
    exp_35: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_106);  sub_106 = None
    sum_36: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
    div_35: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_462: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_35, [0, 2, 3, 1]);  div_35 = None
    permute_463: "f32[16, 16]" = torch.ops.aten.permute.default(arg646_1, [1, 0]);  arg646_1 = None
    clone_496: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
    view_709: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_496, [2654208, 16]);  clone_496 = None
    mm_71: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_709, permute_463);  view_709 = permute_463 = None
    view_710: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_71, [8, 576, 576, 16]);  mm_71 = None
    add_319: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_710, arg647_1);  view_710 = arg647_1 = None
    permute_464: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_319, [0, 3, 1, 2]);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_497: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_464);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_142: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_497, [8, 16, 576, 576]);  clone_497 = None
    clone_498: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_142, memory_format = torch.contiguous_format);  expand_142 = None
    view_711: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_498, [128, 576, 576]);  clone_498 = None
    expand_143: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_107, [8, 16, 576, 48]);  select_107 = None
    clone_499: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
    view_712: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_499, [128, 576, 48]);  clone_499 = None
    bmm_71: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_711, view_712);  view_711 = view_712 = None
    view_713: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_71, [8, 16, 576, 48]);  bmm_71 = None
    permute_465: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_713, [0, 2, 1, 3]);  view_713 = None
    clone_500: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
    view_714: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_500, [8, 576, 768]);  clone_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_715: "f32[4608, 768]" = torch.ops.aten.view.default(view_714, [4608, 768]);  view_714 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(arg648_1, [1, 0]);  arg648_1 = None
    addmm_141: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg649_1, view_715, permute_466);  arg649_1 = view_715 = permute_466 = None
    view_716: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_141, [8, 576, 768]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_501: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_716);  view_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_353: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg71_1, clone_501);  arg71_1 = clone_501 = None
    add_320: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_315, mul_353);  add_315 = mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_502: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_320, memory_format = torch.contiguous_format)
    var_mean_71 = torch.ops.aten.var_mean.correction(clone_502, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 576, 1]" = var_mean_71[0]
    getitem_143: "f32[8, 576, 1]" = var_mean_71[1];  var_mean_71 = None
    add_321: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
    rsqrt_71: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
    sub_107: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_502, getitem_143);  clone_502 = getitem_143 = None
    mul_354: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_71);  sub_107 = rsqrt_71 = None
    mul_355: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_354, arg650_1);  mul_354 = arg650_1 = None
    add_322: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_355, arg651_1);  mul_355 = arg651_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_717: "f32[4608, 768]" = torch.ops.aten.view.default(add_322, [4608, 768]);  add_322 = None
    permute_467: "f32[768, 3072]" = torch.ops.aten.permute.default(arg652_1, [1, 0]);  arg652_1 = None
    addmm_142: "f32[4608, 3072]" = torch.ops.aten.addmm.default(arg653_1, view_717, permute_467);  arg653_1 = view_717 = permute_467 = None
    view_718: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_142, [8, 576, 3072]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_356: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_718, 0.5)
    mul_357: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_718, 0.7071067811865476);  view_718 = None
    erf_35: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_357);  mul_357 = None
    add_323: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_358: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_356, add_323);  mul_356 = add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_503: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_358);  mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_719: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_503, [4608, 3072]);  clone_503 = None
    permute_468: "f32[3072, 768]" = torch.ops.aten.permute.default(arg654_1, [1, 0]);  arg654_1 = None
    addmm_143: "f32[4608, 768]" = torch.ops.aten.addmm.default(arg655_1, view_719, permute_468);  arg655_1 = view_719 = permute_468 = None
    view_720: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_143, [8, 576, 768]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_504: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_720);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_359: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg72_1, clone_504);  arg72_1 = clone_504 = None
    add_324: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_320, mul_359);  add_320 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:347, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand_144: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg73_1, [8, -1, -1]);  arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    cat: "f32[8, 577, 768]" = torch.ops.aten.cat.default([expand_144, add_324], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    var_mean_72 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 577, 1]" = var_mean_72[0]
    getitem_145: "f32[8, 577, 1]" = var_mean_72[1];  var_mean_72 = None
    add_325: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_72: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
    sub_108: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_145);  cat = getitem_145 = None
    mul_360: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_72);  sub_108 = rsqrt_72 = None
    mul_361: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_360, arg656_1);  mul_360 = arg656_1 = None
    add_326: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_361, arg657_1);  mul_361 = arg657_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_1: "f32[8, 577, 768]" = torch.ops.aten.slice.Tensor(add_326, 0, 0, 9223372036854775807)
    select_108: "f32[8, 768]" = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
    permute_469: "f32[768, 768]" = torch.ops.aten.permute.default(arg658_1, [1, 0]);  arg658_1 = None
    addmm_144: "f32[8, 768]" = torch.ops.aten.addmm.default(arg659_1, select_108, permute_469);  arg659_1 = select_108 = permute_469 = None
    unsqueeze: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_144, 1);  addmm_144 = None
    view_721: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze, [8, 1, 16, 48]);  unsqueeze = None
    permute_470: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_721, [0, 2, 1, 3]);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_722: "f32[4616, 768]" = torch.ops.aten.view.default(add_326, [4616, 768])
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(arg660_1, [1, 0]);  arg660_1 = None
    addmm_145: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg661_1, view_722, permute_471);  arg661_1 = view_722 = permute_471 = None
    view_723: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_145, [8, 577, 768]);  addmm_145 = None
    view_724: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_723, [8, 577, 16, 48]);  view_723 = None
    permute_472: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_725: "f32[4616, 768]" = torch.ops.aten.view.default(add_326, [4616, 768]);  add_326 = None
    permute_473: "f32[768, 768]" = torch.ops.aten.permute.default(arg662_1, [1, 0]);  arg662_1 = None
    addmm_146: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg663_1, view_725, permute_473);  arg663_1 = view_725 = permute_473 = None
    view_726: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_146, [8, 577, 768]);  addmm_146 = None
    view_727: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_726, [8, 577, 16, 48]);  view_726 = None
    permute_474: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_470, permute_472, permute_474);  permute_470 = permute_472 = permute_474 = None
    getitem_146: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_475: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    view_728: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_475, [8, 1, 768]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_729: "f32[8, 768]" = torch.ops.aten.view.default(view_728, [8, 768]);  view_728 = None
    permute_476: "f32[768, 768]" = torch.ops.aten.permute.default(arg664_1, [1, 0]);  arg664_1 = None
    addmm_147: "f32[8, 768]" = torch.ops.aten.addmm.default(arg665_1, view_729, permute_476);  arg665_1 = view_729 = permute_476 = None
    view_730: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_147, [8, 1, 768]);  addmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_505: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_730);  view_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_362: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg74_1, clone_505);  arg74_1 = clone_505 = None
    add_327: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(expand_144, mul_362);  expand_144 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    var_mean_73 = torch.ops.aten.var_mean.correction(add_327, [2], correction = 0, keepdim = True)
    getitem_155: "f32[8, 1, 1]" = var_mean_73[0]
    getitem_156: "f32[8, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_328: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_155, 1e-06);  getitem_155 = None
    rsqrt_73: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_328);  add_328 = None
    sub_109: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(add_327, getitem_156);  getitem_156 = None
    mul_363: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_73);  sub_109 = rsqrt_73 = None
    mul_364: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_363, arg666_1);  mul_363 = arg666_1 = None
    add_329: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(mul_364, arg667_1);  mul_364 = arg667_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_731: "f32[8, 768]" = torch.ops.aten.view.default(add_329, [8, 768]);  add_329 = None
    permute_477: "f32[768, 3072]" = torch.ops.aten.permute.default(arg668_1, [1, 0]);  arg668_1 = None
    addmm_148: "f32[8, 3072]" = torch.ops.aten.addmm.default(arg669_1, view_731, permute_477);  arg669_1 = view_731 = permute_477 = None
    view_732: "f32[8, 1, 3072]" = torch.ops.aten.view.default(addmm_148, [8, 1, 3072]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_365: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_732, 0.5)
    mul_366: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_732, 0.7071067811865476);  view_732 = None
    erf_36: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_366);  mul_366 = None
    add_330: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_367: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_365, add_330);  mul_365 = add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_506: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(mul_367);  mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_733: "f32[8, 3072]" = torch.ops.aten.view.default(clone_506, [8, 3072]);  clone_506 = None
    permute_478: "f32[3072, 768]" = torch.ops.aten.permute.default(arg670_1, [1, 0]);  arg670_1 = None
    addmm_149: "f32[8, 768]" = torch.ops.aten.addmm.default(arg671_1, view_733, permute_478);  arg671_1 = view_733 = permute_478 = None
    view_734: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_149, [8, 1, 768]);  addmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_507: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_734);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_368: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg75_1, clone_507);  arg75_1 = clone_507 = None
    add_331: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_327, mul_368);  add_327 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    cat_1: "f32[8, 577, 768]" = torch.ops.aten.cat.default([add_331, add_324], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    var_mean_74 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_157: "f32[8, 577, 1]" = var_mean_74[0]
    getitem_158: "f32[8, 577, 1]" = var_mean_74[1];  var_mean_74 = None
    add_332: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_157, 1e-06);  getitem_157 = None
    rsqrt_74: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_332);  add_332 = None
    sub_110: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_1, getitem_158);  cat_1 = getitem_158 = None
    mul_369: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_74);  sub_110 = rsqrt_74 = None
    mul_370: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_369, arg672_1);  mul_369 = arg672_1 = None
    add_333: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_370, arg673_1);  mul_370 = arg673_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_2: "f32[8, 577, 768]" = torch.ops.aten.slice.Tensor(add_333, 0, 0, 9223372036854775807)
    select_109: "f32[8, 768]" = torch.ops.aten.select.int(slice_2, 1, 0);  slice_2 = None
    permute_479: "f32[768, 768]" = torch.ops.aten.permute.default(arg674_1, [1, 0]);  arg674_1 = None
    addmm_150: "f32[8, 768]" = torch.ops.aten.addmm.default(arg675_1, select_109, permute_479);  arg675_1 = select_109 = permute_479 = None
    unsqueeze_1: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_150, 1);  addmm_150 = None
    view_735: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_1, [8, 1, 16, 48]);  unsqueeze_1 = None
    permute_480: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_736: "f32[4616, 768]" = torch.ops.aten.view.default(add_333, [4616, 768])
    permute_481: "f32[768, 768]" = torch.ops.aten.permute.default(arg676_1, [1, 0]);  arg676_1 = None
    addmm_151: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg677_1, view_736, permute_481);  arg677_1 = view_736 = permute_481 = None
    view_737: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_151, [8, 577, 768]);  addmm_151 = None
    view_738: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_737, [8, 577, 16, 48]);  view_737 = None
    permute_482: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_739: "f32[4616, 768]" = torch.ops.aten.view.default(add_333, [4616, 768]);  add_333 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(arg678_1, [1, 0]);  arg678_1 = None
    addmm_152: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg679_1, view_739, permute_483);  arg679_1 = view_739 = permute_483 = None
    view_740: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_152, [8, 577, 768]);  addmm_152 = None
    view_741: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_740, [8, 577, 16, 48]);  view_740 = None
    permute_484: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_741, [0, 2, 1, 3]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_480, permute_482, permute_484);  permute_480 = permute_482 = permute_484 = None
    getitem_159: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_485: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
    view_742: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_485, [8, 1, 768]);  permute_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_743: "f32[8, 768]" = torch.ops.aten.view.default(view_742, [8, 768]);  view_742 = None
    permute_486: "f32[768, 768]" = torch.ops.aten.permute.default(arg680_1, [1, 0]);  arg680_1 = None
    addmm_153: "f32[8, 768]" = torch.ops.aten.addmm.default(arg681_1, view_743, permute_486);  arg681_1 = view_743 = permute_486 = None
    view_744: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_153, [8, 1, 768]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_508: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_744);  view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_371: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg76_1, clone_508);  arg76_1 = clone_508 = None
    add_334: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_331, mul_371);  add_331 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    var_mean_75 = torch.ops.aten.var_mean.correction(add_334, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 1, 1]" = var_mean_75[0]
    getitem_169: "f32[8, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_335: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_75: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
    sub_111: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(add_334, getitem_169);  getitem_169 = None
    mul_372: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_75);  sub_111 = rsqrt_75 = None
    mul_373: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_372, arg682_1);  mul_372 = arg682_1 = None
    add_336: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(mul_373, arg683_1);  mul_373 = arg683_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_745: "f32[8, 768]" = torch.ops.aten.view.default(add_336, [8, 768]);  add_336 = None
    permute_487: "f32[768, 3072]" = torch.ops.aten.permute.default(arg684_1, [1, 0]);  arg684_1 = None
    addmm_154: "f32[8, 3072]" = torch.ops.aten.addmm.default(arg685_1, view_745, permute_487);  arg685_1 = view_745 = permute_487 = None
    view_746: "f32[8, 1, 3072]" = torch.ops.aten.view.default(addmm_154, [8, 1, 3072]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_374: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_746, 0.5)
    mul_375: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_746, 0.7071067811865476);  view_746 = None
    erf_37: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_337: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_376: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_374, add_337);  mul_374 = add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_509: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(mul_376);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_747: "f32[8, 3072]" = torch.ops.aten.view.default(clone_509, [8, 3072]);  clone_509 = None
    permute_488: "f32[3072, 768]" = torch.ops.aten.permute.default(arg686_1, [1, 0]);  arg686_1 = None
    addmm_155: "f32[8, 768]" = torch.ops.aten.addmm.default(arg687_1, view_747, permute_488);  arg687_1 = view_747 = permute_488 = None
    view_748: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_155, [8, 1, 768]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_510: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_748);  view_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_377: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg77_1, clone_510);  arg77_1 = clone_510 = None
    add_338: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_334, mul_377);  add_334 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:350, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_2: "f32[8, 577, 768]" = torch.ops.aten.cat.default([add_338, add_324], 1);  add_338 = add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:351, code: x = self.norm(x)
    var_mean_76 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_170: "f32[8, 577, 1]" = var_mean_76[0]
    getitem_171: "f32[8, 577, 1]" = var_mean_76[1];  var_mean_76 = None
    add_339: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-06);  getitem_170 = None
    rsqrt_76: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_339);  add_339 = None
    sub_112: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_2, getitem_171);  cat_2 = getitem_171 = None
    mul_378: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_76);  sub_112 = rsqrt_76 = None
    mul_379: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_378, arg688_1);  mul_378 = arg688_1 = None
    add_340: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_379, arg689_1);  mul_379 = arg689_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:356, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_3: "f32[8, 577, 768]" = torch.ops.aten.slice.Tensor(add_340, 0, 0, 9223372036854775807);  add_340 = None
    select_110: "f32[8, 768]" = torch.ops.aten.select.int(slice_3, 1, 0);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:357, code: x = self.head_drop(x)
    clone_511: "f32[8, 768]" = torch.ops.aten.clone.default(select_110);  select_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:358, code: return x if pre_logits else self.head(x)
    permute_489: "f32[768, 1000]" = torch.ops.aten.permute.default(arg690_1, [1, 0]);  arg690_1 = None
    addmm_156: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg691_1, clone_511, permute_489);  arg691_1 = clone_511 = permute_489 = None
    return (addmm_156,)
    