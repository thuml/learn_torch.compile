from __future__ import annotations



def forward(self, arg0_1: "f32[512]", arg1_1: "f32[512]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[128]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[512]", arg17_1: "f32[512]", arg18_1: "f32[128]", arg19_1: "f32[128]", arg20_1: "f32[128]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[128]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[128]", arg35_1: "f32[128]", arg36_1: "f32[128]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[128]", arg47_1: "f32[128]", arg48_1: "f32[512]", arg49_1: "f32[512]", arg50_1: "f32[128]", arg51_1: "f32[128]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[128]", arg55_1: "f32[128]", arg56_1: "f32[128]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[128]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[128]", arg67_1: "f32[128]", arg68_1: "f32[128]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[128]", arg72_1: "f32[128]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[128]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128]", arg87_1: "f32[128]", arg88_1: "f32[128]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[128]", arg92_1: "f32[128]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[512]", arg97_1: "f32[512]", arg98_1: "f32[128]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128]", arg106_1: "f32[128]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[128]", arg111_1: "f32[128]", arg112_1: "f32[512]", arg113_1: "f32[512]", arg114_1: "f32[128]", arg115_1: "f32[128]", arg116_1: "f32[128]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[128]", arg122_1: "f32[128]", arg123_1: "f32[128]", arg124_1: "f32[128]", arg125_1: "f32[128]", arg126_1: "f32[128]", arg127_1: "f32[128]", arg128_1: "f32[512]", arg129_1: "f32[512]", arg130_1: "f32[128]", arg131_1: "f32[128]", arg132_1: "f32[128]", arg133_1: "f32[128]", arg134_1: "f32[128]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[128]", arg138_1: "f32[128]", arg139_1: "f32[128]", arg140_1: "f32[128]", arg141_1: "f32[128]", arg142_1: "f32[128]", arg143_1: "f32[128]", arg144_1: "f32[512]", arg145_1: "f32[512]", arg146_1: "f32[128]", arg147_1: "f32[128]", arg148_1: "f32[128]", arg149_1: "f32[128]", arg150_1: "f32[128]", arg151_1: "f32[128]", arg152_1: "f32[128]", arg153_1: "f32[128]", arg154_1: "f32[128]", arg155_1: "f32[128]", arg156_1: "f32[128]", arg157_1: "f32[128]", arg158_1: "f32[128]", arg159_1: "f32[128]", arg160_1: "f32[512]", arg161_1: "f32[512]", arg162_1: "f32[128]", arg163_1: "f32[128]", arg164_1: "f32[128]", arg165_1: "f32[128]", arg166_1: "f32[128]", arg167_1: "f32[128]", arg168_1: "f32[128]", arg169_1: "f32[128]", arg170_1: "f32[128]", arg171_1: "f32[128]", arg172_1: "f32[128]", arg173_1: "f32[128]", arg174_1: "f32[128]", arg175_1: "f32[128]", arg176_1: "f32[512]", arg177_1: "f32[512]", arg178_1: "f32[128]", arg179_1: "f32[128]", arg180_1: "f32[128]", arg181_1: "f32[128]", arg182_1: "f32[128]", arg183_1: "f32[128]", arg184_1: "f32[128]", arg185_1: "f32[128]", arg186_1: "f32[128]", arg187_1: "f32[128]", arg188_1: "f32[128]", arg189_1: "f32[128]", arg190_1: "f32[128]", arg191_1: "f32[128]", arg192_1: "f32[512]", arg193_1: "f32[512]", arg194_1: "f32[128]", arg195_1: "f32[128]", arg196_1: "f32[128]", arg197_1: "f32[128]", arg198_1: "f32[128]", arg199_1: "f32[128]", arg200_1: "f32[128]", arg201_1: "f32[128]", arg202_1: "f32[128]", arg203_1: "f32[128]", arg204_1: "f32[128]", arg205_1: "f32[128]", arg206_1: "f32[128]", arg207_1: "f32[128]", arg208_1: "f32[512]", arg209_1: "f32[512]", arg210_1: "f32[128]", arg211_1: "f32[128]", arg212_1: "f32[128]", arg213_1: "f32[128]", arg214_1: "f32[128]", arg215_1: "f32[128]", arg216_1: "f32[128]", arg217_1: "f32[128]", arg218_1: "f32[128]", arg219_1: "f32[128]", arg220_1: "f32[128]", arg221_1: "f32[128]", arg222_1: "f32[128]", arg223_1: "f32[128]", arg224_1: "f32[512]", arg225_1: "f32[512]", arg226_1: "f32[128]", arg227_1: "f32[128]", arg228_1: "f32[128]", arg229_1: "f32[128]", arg230_1: "f32[128]", arg231_1: "f32[128]", arg232_1: "f32[128]", arg233_1: "f32[128]", arg234_1: "f32[128]", arg235_1: "f32[128]", arg236_1: "f32[128]", arg237_1: "f32[128]", arg238_1: "f32[128]", arg239_1: "f32[128]", arg240_1: "f32[512]", arg241_1: "f32[512]", arg242_1: "f32[128]", arg243_1: "f32[128]", arg244_1: "f32[128]", arg245_1: "f32[128]", arg246_1: "f32[128]", arg247_1: "f32[128]", arg248_1: "f32[128]", arg249_1: "f32[128]", arg250_1: "f32[128]", arg251_1: "f32[128]", arg252_1: "f32[128]", arg253_1: "f32[128]", arg254_1: "f32[128]", arg255_1: "f32[128]", arg256_1: "f32[512]", arg257_1: "f32[512]", arg258_1: "f32[128]", arg259_1: "f32[128]", arg260_1: "f32[128]", arg261_1: "f32[128]", arg262_1: "f32[128]", arg263_1: "f32[128]", arg264_1: "f32[128]", arg265_1: "f32[128]", arg266_1: "f32[128]", arg267_1: "f32[128]", arg268_1: "f32[128]", arg269_1: "f32[128]", arg270_1: "f32[128]", arg271_1: "f32[128]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[128]", arg275_1: "f32[128]", arg276_1: "f32[128]", arg277_1: "f32[128]", arg278_1: "f32[128]", arg279_1: "f32[128]", arg280_1: "f32[128]", arg281_1: "f32[128]", arg282_1: "f32[128]", arg283_1: "f32[128]", arg284_1: "f32[128]", arg285_1: "f32[128]", arg286_1: "f32[128]", arg287_1: "f32[128]", arg288_1: "f32[512]", arg289_1: "f32[512]", arg290_1: "f32[128]", arg291_1: "f32[128]", arg292_1: "f32[128]", arg293_1: "f32[128]", arg294_1: "f32[128]", arg295_1: "f32[128]", arg296_1: "f32[128]", arg297_1: "f32[128]", arg298_1: "f32[128]", arg299_1: "f32[128]", arg300_1: "f32[128]", arg301_1: "f32[128]", arg302_1: "f32[128]", arg303_1: "f32[128]", arg304_1: "f32[512]", arg305_1: "f32[512]", arg306_1: "f32[128]", arg307_1: "f32[128]", arg308_1: "f32[128]", arg309_1: "f32[128]", arg310_1: "f32[128]", arg311_1: "f32[128]", arg312_1: "f32[128]", arg313_1: "f32[128]", arg314_1: "f32[128]", arg315_1: "f32[128]", arg316_1: "f32[128]", arg317_1: "f32[128]", arg318_1: "f32[128]", arg319_1: "f32[128]", arg320_1: "f32[512]", arg321_1: "f32[512]", arg322_1: "f32[128]", arg323_1: "f32[128]", arg324_1: "f32[128]", arg325_1: "f32[128]", arg326_1: "f32[128]", arg327_1: "f32[128]", arg328_1: "f32[128]", arg329_1: "f32[128]", arg330_1: "f32[128]", arg331_1: "f32[128]", arg332_1: "f32[128]", arg333_1: "f32[128]", arg334_1: "f32[128]", arg335_1: "f32[128]", arg336_1: "f32[512]", arg337_1: "f32[512]", arg338_1: "f32[128]", arg339_1: "f32[128]", arg340_1: "f32[128]", arg341_1: "f32[128]", arg342_1: "f32[128]", arg343_1: "f32[128]", arg344_1: "f32[128]", arg345_1: "f32[128]", arg346_1: "f32[128]", arg347_1: "f32[128]", arg348_1: "f32[128]", arg349_1: "f32[128]", arg350_1: "f32[128]", arg351_1: "f32[128]", arg352_1: "f32[512]", arg353_1: "f32[512]", arg354_1: "f32[128]", arg355_1: "f32[128]", arg356_1: "f32[128]", arg357_1: "f32[128]", arg358_1: "f32[128]", arg359_1: "f32[128]", arg360_1: "f32[128]", arg361_1: "f32[128]", arg362_1: "f32[128]", arg363_1: "f32[128]", arg364_1: "f32[128]", arg365_1: "f32[128]", arg366_1: "f32[128]", arg367_1: "f32[128]", arg368_1: "f32[512]", arg369_1: "f32[512]", arg370_1: "f32[128]", arg371_1: "f32[128]", arg372_1: "f32[128]", arg373_1: "f32[128]", arg374_1: "f32[128]", arg375_1: "f32[128]", arg376_1: "f32[128]", arg377_1: "f32[128]", arg378_1: "f32[128]", arg379_1: "f32[128]", arg380_1: "f32[128]", arg381_1: "f32[128]", arg382_1: "f32[128]", arg383_1: "f32[128]", arg384_1: "f32[512]", arg385_1: "f32[512]", arg386_1: "f32[30522, 128]", arg387_1: "f32[384, 30522]", arg388_1: "f32[30522]", arg389_1: "f32[30522, 128]", arg390_1: "f32[512, 384]", arg391_1: "f32[512]", arg392_1: "f32[512, 512]", arg393_1: "f32[2, 512]", arg394_1: "f32[128, 512]", arg395_1: "f32[128]", arg396_1: "f32[128, 512]", arg397_1: "f32[128]", arg398_1: "f32[128, 128]", arg399_1: "f32[128]", arg400_1: "f32[128, 128]", arg401_1: "f32[128]", arg402_1: "f32[128, 512]", arg403_1: "f32[128]", arg404_1: "f32[128, 128]", arg405_1: "f32[128]", arg406_1: "f32[512, 128]", arg407_1: "f32[512]", arg408_1: "f32[128, 512]", arg409_1: "f32[128]", arg410_1: "f32[512, 128]", arg411_1: "f32[512]", arg412_1: "f32[128, 512]", arg413_1: "f32[128]", arg414_1: "f32[512, 128]", arg415_1: "f32[512]", arg416_1: "f32[128, 512]", arg417_1: "f32[128]", arg418_1: "f32[512, 128]", arg419_1: "f32[512]", arg420_1: "f32[128, 512]", arg421_1: "f32[128]", arg422_1: "f32[512, 128]", arg423_1: "f32[512]", arg424_1: "f32[128, 512]", arg425_1: "f32[128]", arg426_1: "f32[128, 512]", arg427_1: "f32[128]", arg428_1: "f32[128, 128]", arg429_1: "f32[128]", arg430_1: "f32[128, 128]", arg431_1: "f32[128]", arg432_1: "f32[128, 512]", arg433_1: "f32[128]", arg434_1: "f32[128, 128]", arg435_1: "f32[128]", arg436_1: "f32[512, 128]", arg437_1: "f32[512]", arg438_1: "f32[128, 512]", arg439_1: "f32[128]", arg440_1: "f32[512, 128]", arg441_1: "f32[512]", arg442_1: "f32[128, 512]", arg443_1: "f32[128]", arg444_1: "f32[512, 128]", arg445_1: "f32[512]", arg446_1: "f32[128, 512]", arg447_1: "f32[128]", arg448_1: "f32[512, 128]", arg449_1: "f32[512]", arg450_1: "f32[128, 512]", arg451_1: "f32[128]", arg452_1: "f32[512, 128]", arg453_1: "f32[512]", arg454_1: "f32[128, 512]", arg455_1: "f32[128]", arg456_1: "f32[128, 512]", arg457_1: "f32[128]", arg458_1: "f32[128, 128]", arg459_1: "f32[128]", arg460_1: "f32[128, 128]", arg461_1: "f32[128]", arg462_1: "f32[128, 512]", arg463_1: "f32[128]", arg464_1: "f32[128, 128]", arg465_1: "f32[128]", arg466_1: "f32[512, 128]", arg467_1: "f32[512]", arg468_1: "f32[128, 512]", arg469_1: "f32[128]", arg470_1: "f32[512, 128]", arg471_1: "f32[512]", arg472_1: "f32[128, 512]", arg473_1: "f32[128]", arg474_1: "f32[512, 128]", arg475_1: "f32[512]", arg476_1: "f32[128, 512]", arg477_1: "f32[128]", arg478_1: "f32[512, 128]", arg479_1: "f32[512]", arg480_1: "f32[128, 512]", arg481_1: "f32[128]", arg482_1: "f32[512, 128]", arg483_1: "f32[512]", arg484_1: "f32[128, 512]", arg485_1: "f32[128]", arg486_1: "f32[128, 512]", arg487_1: "f32[128]", arg488_1: "f32[128, 128]", arg489_1: "f32[128]", arg490_1: "f32[128, 128]", arg491_1: "f32[128]", arg492_1: "f32[128, 512]", arg493_1: "f32[128]", arg494_1: "f32[128, 128]", arg495_1: "f32[128]", arg496_1: "f32[512, 128]", arg497_1: "f32[512]", arg498_1: "f32[128, 512]", arg499_1: "f32[128]", arg500_1: "f32[512, 128]", arg501_1: "f32[512]", arg502_1: "f32[128, 512]", arg503_1: "f32[128]", arg504_1: "f32[512, 128]", arg505_1: "f32[512]", arg506_1: "f32[128, 512]", arg507_1: "f32[128]", arg508_1: "f32[512, 128]", arg509_1: "f32[512]", arg510_1: "f32[128, 512]", arg511_1: "f32[128]", arg512_1: "f32[512, 128]", arg513_1: "f32[512]", arg514_1: "f32[128, 512]", arg515_1: "f32[128]", arg516_1: "f32[128, 512]", arg517_1: "f32[128]", arg518_1: "f32[128, 128]", arg519_1: "f32[128]", arg520_1: "f32[128, 128]", arg521_1: "f32[128]", arg522_1: "f32[128, 512]", arg523_1: "f32[128]", arg524_1: "f32[128, 128]", arg525_1: "f32[128]", arg526_1: "f32[512, 128]", arg527_1: "f32[512]", arg528_1: "f32[128, 512]", arg529_1: "f32[128]", arg530_1: "f32[512, 128]", arg531_1: "f32[512]", arg532_1: "f32[128, 512]", arg533_1: "f32[128]", arg534_1: "f32[512, 128]", arg535_1: "f32[512]", arg536_1: "f32[128, 512]", arg537_1: "f32[128]", arg538_1: "f32[512, 128]", arg539_1: "f32[512]", arg540_1: "f32[128, 512]", arg541_1: "f32[128]", arg542_1: "f32[512, 128]", arg543_1: "f32[512]", arg544_1: "f32[128, 512]", arg545_1: "f32[128]", arg546_1: "f32[128, 512]", arg547_1: "f32[128]", arg548_1: "f32[128, 128]", arg549_1: "f32[128]", arg550_1: "f32[128, 128]", arg551_1: "f32[128]", arg552_1: "f32[128, 512]", arg553_1: "f32[128]", arg554_1: "f32[128, 128]", arg555_1: "f32[128]", arg556_1: "f32[512, 128]", arg557_1: "f32[512]", arg558_1: "f32[128, 512]", arg559_1: "f32[128]", arg560_1: "f32[512, 128]", arg561_1: "f32[512]", arg562_1: "f32[128, 512]", arg563_1: "f32[128]", arg564_1: "f32[512, 128]", arg565_1: "f32[512]", arg566_1: "f32[128, 512]", arg567_1: "f32[128]", arg568_1: "f32[512, 128]", arg569_1: "f32[512]", arg570_1: "f32[128, 512]", arg571_1: "f32[128]", arg572_1: "f32[512, 128]", arg573_1: "f32[512]", arg574_1: "f32[128, 512]", arg575_1: "f32[128]", arg576_1: "f32[128, 512]", arg577_1: "f32[128]", arg578_1: "f32[128, 128]", arg579_1: "f32[128]", arg580_1: "f32[128, 128]", arg581_1: "f32[128]", arg582_1: "f32[128, 512]", arg583_1: "f32[128]", arg584_1: "f32[128, 128]", arg585_1: "f32[128]", arg586_1: "f32[512, 128]", arg587_1: "f32[512]", arg588_1: "f32[128, 512]", arg589_1: "f32[128]", arg590_1: "f32[512, 128]", arg591_1: "f32[512]", arg592_1: "f32[128, 512]", arg593_1: "f32[128]", arg594_1: "f32[512, 128]", arg595_1: "f32[512]", arg596_1: "f32[128, 512]", arg597_1: "f32[128]", arg598_1: "f32[512, 128]", arg599_1: "f32[512]", arg600_1: "f32[128, 512]", arg601_1: "f32[128]", arg602_1: "f32[512, 128]", arg603_1: "f32[512]", arg604_1: "f32[128, 512]", arg605_1: "f32[128]", arg606_1: "f32[128, 512]", arg607_1: "f32[128]", arg608_1: "f32[128, 128]", arg609_1: "f32[128]", arg610_1: "f32[128, 128]", arg611_1: "f32[128]", arg612_1: "f32[128, 512]", arg613_1: "f32[128]", arg614_1: "f32[128, 128]", arg615_1: "f32[128]", arg616_1: "f32[512, 128]", arg617_1: "f32[512]", arg618_1: "f32[128, 512]", arg619_1: "f32[128]", arg620_1: "f32[512, 128]", arg621_1: "f32[512]", arg622_1: "f32[128, 512]", arg623_1: "f32[128]", arg624_1: "f32[512, 128]", arg625_1: "f32[512]", arg626_1: "f32[128, 512]", arg627_1: "f32[128]", arg628_1: "f32[512, 128]", arg629_1: "f32[512]", arg630_1: "f32[128, 512]", arg631_1: "f32[128]", arg632_1: "f32[512, 128]", arg633_1: "f32[512]", arg634_1: "f32[128, 512]", arg635_1: "f32[128]", arg636_1: "f32[128, 512]", arg637_1: "f32[128]", arg638_1: "f32[128, 128]", arg639_1: "f32[128]", arg640_1: "f32[128, 128]", arg641_1: "f32[128]", arg642_1: "f32[128, 512]", arg643_1: "f32[128]", arg644_1: "f32[128, 128]", arg645_1: "f32[128]", arg646_1: "f32[512, 128]", arg647_1: "f32[512]", arg648_1: "f32[128, 512]", arg649_1: "f32[128]", arg650_1: "f32[512, 128]", arg651_1: "f32[512]", arg652_1: "f32[128, 512]", arg653_1: "f32[128]", arg654_1: "f32[512, 128]", arg655_1: "f32[512]", arg656_1: "f32[128, 512]", arg657_1: "f32[128]", arg658_1: "f32[512, 128]", arg659_1: "f32[512]", arg660_1: "f32[128, 512]", arg661_1: "f32[128]", arg662_1: "f32[512, 128]", arg663_1: "f32[512]", arg664_1: "f32[128, 512]", arg665_1: "f32[128]", arg666_1: "f32[128, 512]", arg667_1: "f32[128]", arg668_1: "f32[128, 128]", arg669_1: "f32[128]", arg670_1: "f32[128, 128]", arg671_1: "f32[128]", arg672_1: "f32[128, 512]", arg673_1: "f32[128]", arg674_1: "f32[128, 128]", arg675_1: "f32[128]", arg676_1: "f32[512, 128]", arg677_1: "f32[512]", arg678_1: "f32[128, 512]", arg679_1: "f32[128]", arg680_1: "f32[512, 128]", arg681_1: "f32[512]", arg682_1: "f32[128, 512]", arg683_1: "f32[128]", arg684_1: "f32[512, 128]", arg685_1: "f32[512]", arg686_1: "f32[128, 512]", arg687_1: "f32[128]", arg688_1: "f32[512, 128]", arg689_1: "f32[512]", arg690_1: "f32[128, 512]", arg691_1: "f32[128]", arg692_1: "f32[512, 128]", arg693_1: "f32[512]", arg694_1: "f32[128, 512]", arg695_1: "f32[128]", arg696_1: "f32[128, 512]", arg697_1: "f32[128]", arg698_1: "f32[128, 128]", arg699_1: "f32[128]", arg700_1: "f32[128, 128]", arg701_1: "f32[128]", arg702_1: "f32[128, 512]", arg703_1: "f32[128]", arg704_1: "f32[128, 128]", arg705_1: "f32[128]", arg706_1: "f32[512, 128]", arg707_1: "f32[512]", arg708_1: "f32[128, 512]", arg709_1: "f32[128]", arg710_1: "f32[512, 128]", arg711_1: "f32[512]", arg712_1: "f32[128, 512]", arg713_1: "f32[128]", arg714_1: "f32[512, 128]", arg715_1: "f32[512]", arg716_1: "f32[128, 512]", arg717_1: "f32[128]", arg718_1: "f32[512, 128]", arg719_1: "f32[512]", arg720_1: "f32[128, 512]", arg721_1: "f32[128]", arg722_1: "f32[512, 128]", arg723_1: "f32[512]", arg724_1: "f32[128, 512]", arg725_1: "f32[128]", arg726_1: "f32[128, 512]", arg727_1: "f32[128]", arg728_1: "f32[128, 128]", arg729_1: "f32[128]", arg730_1: "f32[128, 128]", arg731_1: "f32[128]", arg732_1: "f32[128, 512]", arg733_1: "f32[128]", arg734_1: "f32[128, 128]", arg735_1: "f32[128]", arg736_1: "f32[512, 128]", arg737_1: "f32[512]", arg738_1: "f32[128, 512]", arg739_1: "f32[128]", arg740_1: "f32[512, 128]", arg741_1: "f32[512]", arg742_1: "f32[128, 512]", arg743_1: "f32[128]", arg744_1: "f32[512, 128]", arg745_1: "f32[512]", arg746_1: "f32[128, 512]", arg747_1: "f32[128]", arg748_1: "f32[512, 128]", arg749_1: "f32[512]", arg750_1: "f32[128, 512]", arg751_1: "f32[128]", arg752_1: "f32[512, 128]", arg753_1: "f32[512]", arg754_1: "f32[128, 512]", arg755_1: "f32[128]", arg756_1: "f32[128, 512]", arg757_1: "f32[128]", arg758_1: "f32[128, 128]", arg759_1: "f32[128]", arg760_1: "f32[128, 128]", arg761_1: "f32[128]", arg762_1: "f32[128, 512]", arg763_1: "f32[128]", arg764_1: "f32[128, 128]", arg765_1: "f32[128]", arg766_1: "f32[512, 128]", arg767_1: "f32[512]", arg768_1: "f32[128, 512]", arg769_1: "f32[128]", arg770_1: "f32[512, 128]", arg771_1: "f32[512]", arg772_1: "f32[128, 512]", arg773_1: "f32[128]", arg774_1: "f32[512, 128]", arg775_1: "f32[512]", arg776_1: "f32[128, 512]", arg777_1: "f32[128]", arg778_1: "f32[512, 128]", arg779_1: "f32[512]", arg780_1: "f32[128, 512]", arg781_1: "f32[128]", arg782_1: "f32[512, 128]", arg783_1: "f32[512]", arg784_1: "f32[128, 512]", arg785_1: "f32[128]", arg786_1: "f32[128, 512]", arg787_1: "f32[128]", arg788_1: "f32[128, 128]", arg789_1: "f32[128]", arg790_1: "f32[128, 128]", arg791_1: "f32[128]", arg792_1: "f32[128, 512]", arg793_1: "f32[128]", arg794_1: "f32[128, 128]", arg795_1: "f32[128]", arg796_1: "f32[512, 128]", arg797_1: "f32[512]", arg798_1: "f32[128, 512]", arg799_1: "f32[128]", arg800_1: "f32[512, 128]", arg801_1: "f32[512]", arg802_1: "f32[128, 512]", arg803_1: "f32[128]", arg804_1: "f32[512, 128]", arg805_1: "f32[512]", arg806_1: "f32[128, 512]", arg807_1: "f32[128]", arg808_1: "f32[512, 128]", arg809_1: "f32[512]", arg810_1: "f32[128, 512]", arg811_1: "f32[128]", arg812_1: "f32[512, 128]", arg813_1: "f32[512]", arg814_1: "f32[128, 512]", arg815_1: "f32[128]", arg816_1: "f32[128, 512]", arg817_1: "f32[128]", arg818_1: "f32[128, 128]", arg819_1: "f32[128]", arg820_1: "f32[128, 128]", arg821_1: "f32[128]", arg822_1: "f32[128, 512]", arg823_1: "f32[128]", arg824_1: "f32[128, 128]", arg825_1: "f32[128]", arg826_1: "f32[512, 128]", arg827_1: "f32[512]", arg828_1: "f32[128, 512]", arg829_1: "f32[128]", arg830_1: "f32[512, 128]", arg831_1: "f32[512]", arg832_1: "f32[128, 512]", arg833_1: "f32[128]", arg834_1: "f32[512, 128]", arg835_1: "f32[512]", arg836_1: "f32[128, 512]", arg837_1: "f32[128]", arg838_1: "f32[512, 128]", arg839_1: "f32[512]", arg840_1: "f32[128, 512]", arg841_1: "f32[128]", arg842_1: "f32[512, 128]", arg843_1: "f32[512]", arg844_1: "f32[128, 512]", arg845_1: "f32[128]", arg846_1: "f32[128, 512]", arg847_1: "f32[128]", arg848_1: "f32[128, 128]", arg849_1: "f32[128]", arg850_1: "f32[128, 128]", arg851_1: "f32[128]", arg852_1: "f32[128, 512]", arg853_1: "f32[128]", arg854_1: "f32[128, 128]", arg855_1: "f32[128]", arg856_1: "f32[512, 128]", arg857_1: "f32[512]", arg858_1: "f32[128, 512]", arg859_1: "f32[128]", arg860_1: "f32[512, 128]", arg861_1: "f32[512]", arg862_1: "f32[128, 512]", arg863_1: "f32[128]", arg864_1: "f32[512, 128]", arg865_1: "f32[512]", arg866_1: "f32[128, 512]", arg867_1: "f32[128]", arg868_1: "f32[512, 128]", arg869_1: "f32[512]", arg870_1: "f32[128, 512]", arg871_1: "f32[128]", arg872_1: "f32[512, 128]", arg873_1: "f32[512]", arg874_1: "f32[128, 512]", arg875_1: "f32[128]", arg876_1: "f32[128, 512]", arg877_1: "f32[128]", arg878_1: "f32[128, 128]", arg879_1: "f32[128]", arg880_1: "f32[128, 128]", arg881_1: "f32[128]", arg882_1: "f32[128, 512]", arg883_1: "f32[128]", arg884_1: "f32[128, 128]", arg885_1: "f32[128]", arg886_1: "f32[512, 128]", arg887_1: "f32[512]", arg888_1: "f32[128, 512]", arg889_1: "f32[128]", arg890_1: "f32[512, 128]", arg891_1: "f32[512]", arg892_1: "f32[128, 512]", arg893_1: "f32[128]", arg894_1: "f32[512, 128]", arg895_1: "f32[512]", arg896_1: "f32[128, 512]", arg897_1: "f32[128]", arg898_1: "f32[512, 128]", arg899_1: "f32[512]", arg900_1: "f32[128, 512]", arg901_1: "f32[128]", arg902_1: "f32[512, 128]", arg903_1: "f32[512]", arg904_1: "f32[128, 512]", arg905_1: "f32[128]", arg906_1: "f32[128, 512]", arg907_1: "f32[128]", arg908_1: "f32[128, 128]", arg909_1: "f32[128]", arg910_1: "f32[128, 128]", arg911_1: "f32[128]", arg912_1: "f32[128, 512]", arg913_1: "f32[128]", arg914_1: "f32[128, 128]", arg915_1: "f32[128]", arg916_1: "f32[512, 128]", arg917_1: "f32[512]", arg918_1: "f32[128, 512]", arg919_1: "f32[128]", arg920_1: "f32[512, 128]", arg921_1: "f32[512]", arg922_1: "f32[128, 512]", arg923_1: "f32[128]", arg924_1: "f32[512, 128]", arg925_1: "f32[512]", arg926_1: "f32[128, 512]", arg927_1: "f32[128]", arg928_1: "f32[512, 128]", arg929_1: "f32[512]", arg930_1: "f32[128, 512]", arg931_1: "f32[128]", arg932_1: "f32[512, 128]", arg933_1: "f32[512]", arg934_1: "f32[128, 512]", arg935_1: "f32[128]", arg936_1: "f32[128, 512]", arg937_1: "f32[128]", arg938_1: "f32[128, 128]", arg939_1: "f32[128]", arg940_1: "f32[128, 128]", arg941_1: "f32[128]", arg942_1: "f32[128, 512]", arg943_1: "f32[128]", arg944_1: "f32[128, 128]", arg945_1: "f32[128]", arg946_1: "f32[512, 128]", arg947_1: "f32[512]", arg948_1: "f32[128, 512]", arg949_1: "f32[128]", arg950_1: "f32[512, 128]", arg951_1: "f32[512]", arg952_1: "f32[128, 512]", arg953_1: "f32[128]", arg954_1: "f32[512, 128]", arg955_1: "f32[512]", arg956_1: "f32[128, 512]", arg957_1: "f32[128]", arg958_1: "f32[512, 128]", arg959_1: "f32[512]", arg960_1: "f32[128, 512]", arg961_1: "f32[128]", arg962_1: "f32[512, 128]", arg963_1: "f32[512]", arg964_1: "f32[128, 512]", arg965_1: "f32[128]", arg966_1: "f32[128, 512]", arg967_1: "f32[128]", arg968_1: "f32[128, 128]", arg969_1: "f32[128]", arg970_1: "f32[128, 128]", arg971_1: "f32[128]", arg972_1: "f32[128, 512]", arg973_1: "f32[128]", arg974_1: "f32[128, 128]", arg975_1: "f32[128]", arg976_1: "f32[512, 128]", arg977_1: "f32[512]", arg978_1: "f32[128, 512]", arg979_1: "f32[128]", arg980_1: "f32[512, 128]", arg981_1: "f32[512]", arg982_1: "f32[128, 512]", arg983_1: "f32[128]", arg984_1: "f32[512, 128]", arg985_1: "f32[512]", arg986_1: "f32[128, 512]", arg987_1: "f32[128]", arg988_1: "f32[512, 128]", arg989_1: "f32[512]", arg990_1: "f32[128, 512]", arg991_1: "f32[128]", arg992_1: "f32[512, 128]", arg993_1: "f32[512]", arg994_1: "f32[128, 512]", arg995_1: "f32[128]", arg996_1: "f32[128, 512]", arg997_1: "f32[128]", arg998_1: "f32[128, 128]", arg999_1: "f32[128]", arg1000_1: "f32[128, 128]", arg1001_1: "f32[128]", arg1002_1: "f32[128, 512]", arg1003_1: "f32[128]", arg1004_1: "f32[128, 128]", arg1005_1: "f32[128]", arg1006_1: "f32[512, 128]", arg1007_1: "f32[512]", arg1008_1: "f32[128, 512]", arg1009_1: "f32[128]", arg1010_1: "f32[512, 128]", arg1011_1: "f32[512]", arg1012_1: "f32[128, 512]", arg1013_1: "f32[128]", arg1014_1: "f32[512, 128]", arg1015_1: "f32[512]", arg1016_1: "f32[128, 512]", arg1017_1: "f32[128]", arg1018_1: "f32[512, 128]", arg1019_1: "f32[512]", arg1020_1: "f32[128, 512]", arg1021_1: "f32[128]", arg1022_1: "f32[512, 128]", arg1023_1: "f32[512]", arg1024_1: "f32[128, 512]", arg1025_1: "f32[128]", arg1026_1: "f32[128, 512]", arg1027_1: "f32[128]", arg1028_1: "f32[128, 128]", arg1029_1: "f32[128]", arg1030_1: "f32[128, 128]", arg1031_1: "f32[128]", arg1032_1: "f32[128, 512]", arg1033_1: "f32[128]", arg1034_1: "f32[128, 128]", arg1035_1: "f32[128]", arg1036_1: "f32[512, 128]", arg1037_1: "f32[512]", arg1038_1: "f32[128, 512]", arg1039_1: "f32[128]", arg1040_1: "f32[512, 128]", arg1041_1: "f32[512]", arg1042_1: "f32[128, 512]", arg1043_1: "f32[128]", arg1044_1: "f32[512, 128]", arg1045_1: "f32[512]", arg1046_1: "f32[128, 512]", arg1047_1: "f32[128]", arg1048_1: "f32[512, 128]", arg1049_1: "f32[512]", arg1050_1: "f32[128, 512]", arg1051_1: "f32[128]", arg1052_1: "f32[512, 128]", arg1053_1: "f32[512]", arg1054_1: "f32[128, 512]", arg1055_1: "f32[128]", arg1056_1: "f32[128, 512]", arg1057_1: "f32[128]", arg1058_1: "f32[128, 128]", arg1059_1: "f32[128]", arg1060_1: "f32[128, 128]", arg1061_1: "f32[128]", arg1062_1: "f32[128, 512]", arg1063_1: "f32[128]", arg1064_1: "f32[128, 128]", arg1065_1: "f32[128]", arg1066_1: "f32[512, 128]", arg1067_1: "f32[512]", arg1068_1: "f32[128, 512]", arg1069_1: "f32[128]", arg1070_1: "f32[512, 128]", arg1071_1: "f32[512]", arg1072_1: "f32[128, 512]", arg1073_1: "f32[128]", arg1074_1: "f32[512, 128]", arg1075_1: "f32[512]", arg1076_1: "f32[128, 512]", arg1077_1: "f32[128]", arg1078_1: "f32[512, 128]", arg1079_1: "f32[512]", arg1080_1: "f32[128, 512]", arg1081_1: "f32[128]", arg1082_1: "f32[512, 128]", arg1083_1: "f32[512]", arg1084_1: "f32[128, 512]", arg1085_1: "f32[128]", arg1086_1: "f32[128, 512]", arg1087_1: "f32[128]", arg1088_1: "f32[128, 128]", arg1089_1: "f32[128]", arg1090_1: "f32[128, 128]", arg1091_1: "f32[128]", arg1092_1: "f32[128, 512]", arg1093_1: "f32[128]", arg1094_1: "f32[128, 128]", arg1095_1: "f32[128]", arg1096_1: "f32[512, 128]", arg1097_1: "f32[512]", arg1098_1: "f32[128, 512]", arg1099_1: "f32[128]", arg1100_1: "f32[512, 128]", arg1101_1: "f32[512]", arg1102_1: "f32[128, 512]", arg1103_1: "f32[128]", arg1104_1: "f32[512, 128]", arg1105_1: "f32[512]", arg1106_1: "f32[128, 512]", arg1107_1: "f32[128]", arg1108_1: "f32[512, 128]", arg1109_1: "f32[512]", arg1110_1: "f32[128, 512]", arg1111_1: "f32[128]", arg1112_1: "f32[512, 128]", arg1113_1: "f32[512]", arg1114_1: "f32[512, 512]", arg1115_1: "f32[512]", arg1116_1: "f32[512]", arg1117_1: "f32[512]", arg1118_1: "i64[1, 512]", arg1119_1: "i64[1, 128]", arg1120_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:880, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    unsqueeze: "f32[1, 1, 128]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 128]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    full_default_1: "f32[1, 1, 1, 128]" = torch.ops.aten.full.default([1, 1, 1, 128], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:218, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 128, 128]" = torch.ops.aten.embedding.default(arg389_1, arg1119_1, 0);  arg389_1 = arg1119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:230, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
    slice_6: "f32[1, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)
    constant_pad_nd: "f32[1, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_6, [0, 0, 0, 1, 0, 0], 0.0);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:232, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
    slice_8: "f32[1, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)
    constant_pad_nd_1: "f32[1, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_8, [0, 0, 1, 0, 0, 0], 0.0);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:228, code: inputs_embeds = torch.cat(
    cat: "f32[1, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:237, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
    view: "f32[128, 384]" = torch.ops.aten.reshape.default(cat, [128, 384]);  cat = None
    permute: "f32[384, 512]" = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
    
    # No stacktrace found for following nodes
    mm_default_361: "f32[128, 512]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
    add_tensor_361: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_361, arg391_1);  mm_default_361 = arg391_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:237, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
    view_1: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_361, [1, 128, 512]);  add_tensor_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:213, code: position_ids = self.position_ids[:, :seq_length]
    slice_4: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1118_1, 1, 0, 128);  arg1118_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:241, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg392_1, slice_4);  arg392_1 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:243, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:882, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_default: "i64[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:242, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_2: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg393_1, full_default);  arg393_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:243, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add_1: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
    add_2: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_4: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_2: "f32[512, 128]" = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
    
    # No stacktrace found for following nodes
    mm_default_360: "f32[128, 128]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
    add_tensor_360: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_360, arg397_1);  mm_default_360 = arg397_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_5: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_360, [1, 128, 128]);  add_tensor_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_3: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_5, arg4_1);  view_5 = arg4_1 = None
    add_4: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_6: "f32[128, 128]" = torch.ops.aten.reshape.default(add_4, [128, 128])
    permute_3: "f32[128, 128]" = torch.ops.aten.permute.default(arg398_1, [1, 0]);  arg398_1 = None
    
    # No stacktrace found for following nodes
    mm_default_359: "f32[128, 128]" = torch.ops.aten.mm.default(view_6, permute_3);  view_6 = permute_3 = None
    add_tensor_359: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_359, arg399_1);  mm_default_359 = arg399_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_7: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_359, [1, 128, 128]);  add_tensor_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_12: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_7, [1, 128, 4, 32]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    
    # No stacktrace found for following nodes
    clone_default_69: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_8: "f32[128, 128]" = torch.ops.aten.reshape.default(add_4, [128, 128]);  add_4 = None
    permute_4: "f32[128, 128]" = torch.ops.aten.permute.default(arg400_1, [1, 0]);  arg400_1 = None
    
    # No stacktrace found for following nodes
    mm_default_358: "f32[128, 128]" = torch.ops.aten.mm.default(view_8, permute_4);  view_8 = permute_4 = None
    add_tensor_358: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_358, arg401_1);  mm_default_358 = arg401_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_9: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_358, [1, 128, 128]);  add_tensor_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_13: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_9, [1, 128, 4, 32]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_7: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # No stacktrace found for following nodes
    clone_default_70: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_10: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_5: "f32[512, 128]" = torch.ops.aten.permute.default(arg402_1, [1, 0]);  arg402_1 = None
    
    # No stacktrace found for following nodes
    mm_default_357: "f32[128, 128]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
    add_tensor_357: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_357, arg403_1);  mm_default_357 = arg403_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_11: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_357, [1, 128, 128]);  add_tensor_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_14: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_11, [1, 128, 4, 32]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_8: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # No stacktrace found for following nodes
    clone_default_71: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_69, clone_default_70, clone_default_71, None, False, scale = 0.17677669529663687);  clone_default_69 = clone_default_70 = clone_default_71 = None
    getitem_25: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_23[0];  _scaled_dot_product_efficient_attention_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_10: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_25, [0, 2, 1, 3]);  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_21: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_10, [1, 128, 128]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_22: "f32[128, 128]" = torch.ops.aten.reshape.default(view_21, [128, 128]);  view_21 = None
    permute_11: "f32[128, 128]" = torch.ops.aten.permute.default(arg404_1, [1, 0]);  arg404_1 = None
    
    # No stacktrace found for following nodes
    mm_default_356: "f32[128, 128]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
    add_tensor_356: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_356, arg405_1);  mm_default_356 = arg405_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_23: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_356, [1, 128, 128]);  add_tensor_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_2: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_1: "f32[512, 128]" = torch.ops.aten.permute.default(arg394_1, [1, 0]);  arg394_1 = None
    
    # No stacktrace found for following nodes
    mm_default_355: "f32[128, 128]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
    add_tensor_355: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_355, arg395_1);  mm_default_355 = arg395_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_3: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_355, [1, 128, 128]);  add_tensor_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_2: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_3, arg2_1);  view_3 = arg2_1 = None
    add_3: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_6: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_23, add_3);  view_23 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_4: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_6, arg6_1);  add_6 = arg6_1 = None
    add_7: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_4, arg7_1);  mul_4 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_24: "f32[128, 128]" = torch.ops.aten.reshape.default(add_7, [128, 128])
    permute_12: "f32[128, 512]" = torch.ops.aten.permute.default(arg406_1, [1, 0]);  arg406_1 = None
    
    # No stacktrace found for following nodes
    mm_default_354: "f32[128, 512]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
    add_tensor_354: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_354, arg407_1);  mm_default_354 = arg407_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_25: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_354, [1, 128, 512]);  add_tensor_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_25);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_26: "f32[128, 512]" = torch.ops.aten.reshape.default(relu, [128, 512]);  relu = None
    permute_13: "f32[512, 128]" = torch.ops.aten.permute.default(arg408_1, [1, 0]);  arg408_1 = None
    
    # No stacktrace found for following nodes
    mm_default_353: "f32[128, 128]" = torch.ops.aten.mm.default(view_26, permute_13);  view_26 = permute_13 = None
    add_tensor_353: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_353, arg409_1);  mm_default_353 = arg409_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_27: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_353, [1, 128, 128]);  add_tensor_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_8: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_27, add_7);  view_27 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_5: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_8, arg8_1);  add_8 = arg8_1 = None
    add_9: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_5, arg9_1);  mul_5 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_28: "f32[128, 128]" = torch.ops.aten.reshape.default(add_9, [128, 128])
    permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(arg410_1, [1, 0]);  arg410_1 = None
    
    # No stacktrace found for following nodes
    mm_default_352: "f32[128, 512]" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
    add_tensor_352: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_352, arg411_1);  mm_default_352 = arg411_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_29: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_352, [1, 128, 512]);  add_tensor_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_1: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_29);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_30: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_1, [128, 512]);  relu_1 = None
    permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(arg412_1, [1, 0]);  arg412_1 = None
    
    # No stacktrace found for following nodes
    mm_default_351: "f32[128, 128]" = torch.ops.aten.mm.default(view_30, permute_15);  view_30 = permute_15 = None
    add_tensor_351: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_351, arg413_1);  mm_default_351 = arg413_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_31: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_351, [1, 128, 128]);  add_tensor_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_10: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_31, add_9);  view_31 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_6: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_10, arg10_1);  add_10 = arg10_1 = None
    add_11: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, arg11_1);  mul_6 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[128, 128]" = torch.ops.aten.reshape.default(add_11, [128, 128])
    permute_16: "f32[128, 512]" = torch.ops.aten.permute.default(arg414_1, [1, 0]);  arg414_1 = None
    
    # No stacktrace found for following nodes
    mm_default_350: "f32[128, 512]" = torch.ops.aten.mm.default(view_32, permute_16);  view_32 = permute_16 = None
    add_tensor_350: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_350, arg415_1);  mm_default_350 = arg415_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_33: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_350, [1, 128, 512]);  add_tensor_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_2: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_34: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_2, [128, 512]);  relu_2 = None
    permute_17: "f32[512, 128]" = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
    
    # No stacktrace found for following nodes
    mm_default_349: "f32[128, 128]" = torch.ops.aten.mm.default(view_34, permute_17);  view_34 = permute_17 = None
    add_tensor_349: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_349, arg417_1);  mm_default_349 = arg417_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_35: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_349, [1, 128, 128]);  add_tensor_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_12: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_35, add_11);  view_35 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_7: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_12, arg12_1);  add_12 = arg12_1 = None
    add_13: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_7, arg13_1);  mul_7 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_36: "f32[128, 128]" = torch.ops.aten.reshape.default(add_13, [128, 128])
    permute_18: "f32[128, 512]" = torch.ops.aten.permute.default(arg418_1, [1, 0]);  arg418_1 = None
    
    # No stacktrace found for following nodes
    mm_default_348: "f32[128, 512]" = torch.ops.aten.mm.default(view_36, permute_18);  view_36 = permute_18 = None
    add_tensor_348: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_348, arg419_1);  mm_default_348 = arg419_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_37: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_348, [1, 128, 512]);  add_tensor_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_3: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_37);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_38: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_3, [128, 512]);  relu_3 = None
    permute_19: "f32[512, 128]" = torch.ops.aten.permute.default(arg420_1, [1, 0]);  arg420_1 = None
    
    # No stacktrace found for following nodes
    mm_default_347: "f32[128, 128]" = torch.ops.aten.mm.default(view_38, permute_19);  view_38 = permute_19 = None
    add_tensor_347: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_347, arg421_1);  mm_default_347 = arg421_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_39: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_347, [1, 128, 128]);  add_tensor_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_14: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_39, add_13);  view_39 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_8: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, arg14_1);  add_14 = arg14_1 = None
    add_15: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_40: "f32[128, 128]" = torch.ops.aten.reshape.default(add_15, [128, 128]);  add_15 = None
    permute_20: "f32[128, 512]" = torch.ops.aten.permute.default(arg422_1, [1, 0]);  arg422_1 = None
    
    # No stacktrace found for following nodes
    mm_default_346: "f32[128, 512]" = torch.ops.aten.mm.default(view_40, permute_20);  view_40 = permute_20 = None
    add_tensor_346: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_346, arg423_1);  mm_default_346 = arg423_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_41: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_346, [1, 128, 512]);  add_tensor_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_16: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_41, add_2);  view_41 = add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_16, arg16_1);  add_16 = arg16_1 = None
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_9, arg17_1);  mul_9 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_44: "f32[128, 512]" = torch.ops.aten.reshape.default(add_17, [128, 512])
    permute_22: "f32[512, 128]" = torch.ops.aten.permute.default(arg426_1, [1, 0]);  arg426_1 = None
    
    # No stacktrace found for following nodes
    mm_default_345: "f32[128, 128]" = torch.ops.aten.mm.default(view_44, permute_22);  view_44 = permute_22 = None
    add_tensor_345: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_345, arg427_1);  mm_default_345 = arg427_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_45: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_345, [1, 128, 128]);  add_tensor_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_11: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_45, arg20_1);  view_45 = arg20_1 = None
    add_19: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_46: "f32[128, 128]" = torch.ops.aten.reshape.default(add_19, [128, 128])
    permute_23: "f32[128, 128]" = torch.ops.aten.permute.default(arg428_1, [1, 0]);  arg428_1 = None
    
    # No stacktrace found for following nodes
    mm_default_344: "f32[128, 128]" = torch.ops.aten.mm.default(view_46, permute_23);  view_46 = permute_23 = None
    add_tensor_344: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_344, arg429_1);  mm_default_344 = arg429_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_47: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_344, [1, 128, 128]);  add_tensor_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_52: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_47, [1, 128, 4, 32]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # No stacktrace found for following nodes
    clone_default_66: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_48: "f32[128, 128]" = torch.ops.aten.reshape.default(add_19, [128, 128]);  add_19 = None
    permute_24: "f32[128, 128]" = torch.ops.aten.permute.default(arg430_1, [1, 0]);  arg430_1 = None
    
    # No stacktrace found for following nodes
    mm_default_343: "f32[128, 128]" = torch.ops.aten.mm.default(view_48, permute_24);  view_48 = permute_24 = None
    add_tensor_343: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_343, arg431_1);  mm_default_343 = arg431_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_49: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_343, [1, 128, 128]);  add_tensor_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_53: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_49, [1, 128, 4, 32]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # No stacktrace found for following nodes
    clone_default_67: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_50: "f32[128, 512]" = torch.ops.aten.reshape.default(add_17, [128, 512])
    permute_25: "f32[512, 128]" = torch.ops.aten.permute.default(arg432_1, [1, 0]);  arg432_1 = None
    
    # No stacktrace found for following nodes
    mm_default_342: "f32[128, 128]" = torch.ops.aten.mm.default(view_50, permute_25);  view_50 = permute_25 = None
    add_tensor_342: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_342, arg433_1);  mm_default_342 = arg433_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_51: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_342, [1, 128, 128]);  add_tensor_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_54: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_51, [1, 128, 4, 32]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # No stacktrace found for following nodes
    clone_default_68: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_66, clone_default_67, clone_default_68, None, False, scale = 0.17677669529663687);  clone_default_66 = clone_default_67 = clone_default_68 = None
    getitem_24: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_22[0];  _scaled_dot_product_efficient_attention_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_30: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_24, [0, 2, 1, 3]);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_61: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_30, [1, 128, 128]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_62: "f32[128, 128]" = torch.ops.aten.reshape.default(view_61, [128, 128]);  view_61 = None
    permute_31: "f32[128, 128]" = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
    
    # No stacktrace found for following nodes
    mm_default_341: "f32[128, 128]" = torch.ops.aten.mm.default(view_62, permute_31);  view_62 = permute_31 = None
    add_tensor_341: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_341, arg435_1);  mm_default_341 = arg435_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_63: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_341, [1, 128, 128]);  add_tensor_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_42: "f32[128, 512]" = torch.ops.aten.reshape.default(add_17, [128, 512])
    permute_21: "f32[512, 128]" = torch.ops.aten.permute.default(arg424_1, [1, 0]);  arg424_1 = None
    
    # No stacktrace found for following nodes
    mm_default_340: "f32[128, 128]" = torch.ops.aten.mm.default(view_42, permute_21);  view_42 = permute_21 = None
    add_tensor_340: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_340, arg425_1);  mm_default_340 = arg425_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_43: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_340, [1, 128, 128]);  add_tensor_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_10: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_43, arg18_1);  view_43 = arg18_1 = None
    add_18: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, arg19_1);  mul_10 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_21: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_63, add_18);  view_63 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_12: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_21, arg22_1);  add_21 = arg22_1 = None
    add_22: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_12, arg23_1);  mul_12 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[128, 128]" = torch.ops.aten.reshape.default(add_22, [128, 128])
    permute_32: "f32[128, 512]" = torch.ops.aten.permute.default(arg436_1, [1, 0]);  arg436_1 = None
    
    # No stacktrace found for following nodes
    mm_default_339: "f32[128, 512]" = torch.ops.aten.mm.default(view_64, permute_32);  view_64 = permute_32 = None
    add_tensor_339: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_339, arg437_1);  mm_default_339 = arg437_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_65: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_339, [1, 128, 512]);  add_tensor_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_4: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_66: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_4, [128, 512]);  relu_4 = None
    permute_33: "f32[512, 128]" = torch.ops.aten.permute.default(arg438_1, [1, 0]);  arg438_1 = None
    
    # No stacktrace found for following nodes
    mm_default_338: "f32[128, 128]" = torch.ops.aten.mm.default(view_66, permute_33);  view_66 = permute_33 = None
    add_tensor_338: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_338, arg439_1);  mm_default_338 = arg439_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_67: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_338, [1, 128, 128]);  add_tensor_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_23: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_67, add_22);  view_67 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_13: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_23, arg24_1);  add_23 = arg24_1 = None
    add_24: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, arg25_1);  mul_13 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[128, 128]" = torch.ops.aten.reshape.default(add_24, [128, 128])
    permute_34: "f32[128, 512]" = torch.ops.aten.permute.default(arg440_1, [1, 0]);  arg440_1 = None
    
    # No stacktrace found for following nodes
    mm_default_337: "f32[128, 512]" = torch.ops.aten.mm.default(view_68, permute_34);  view_68 = permute_34 = None
    add_tensor_337: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_337, arg441_1);  mm_default_337 = arg441_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_69: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_337, [1, 128, 512]);  add_tensor_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_5: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_69);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_70: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_5, [128, 512]);  relu_5 = None
    permute_35: "f32[512, 128]" = torch.ops.aten.permute.default(arg442_1, [1, 0]);  arg442_1 = None
    
    # No stacktrace found for following nodes
    mm_default_336: "f32[128, 128]" = torch.ops.aten.mm.default(view_70, permute_35);  view_70 = permute_35 = None
    add_tensor_336: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_336, arg443_1);  mm_default_336 = arg443_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_71: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_336, [1, 128, 128]);  add_tensor_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_25: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_71, add_24);  view_71 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_14: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, arg26_1);  add_25 = arg26_1 = None
    add_26: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, arg27_1);  mul_14 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_72: "f32[128, 128]" = torch.ops.aten.reshape.default(add_26, [128, 128])
    permute_36: "f32[128, 512]" = torch.ops.aten.permute.default(arg444_1, [1, 0]);  arg444_1 = None
    
    # No stacktrace found for following nodes
    mm_default_335: "f32[128, 512]" = torch.ops.aten.mm.default(view_72, permute_36);  view_72 = permute_36 = None
    add_tensor_335: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_335, arg445_1);  mm_default_335 = arg445_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_73: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_335, [1, 128, 512]);  add_tensor_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_6: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_74: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_6, [128, 512]);  relu_6 = None
    permute_37: "f32[512, 128]" = torch.ops.aten.permute.default(arg446_1, [1, 0]);  arg446_1 = None
    
    # No stacktrace found for following nodes
    mm_default_334: "f32[128, 128]" = torch.ops.aten.mm.default(view_74, permute_37);  view_74 = permute_37 = None
    add_tensor_334: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_334, arg447_1);  mm_default_334 = arg447_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_75: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_334, [1, 128, 128]);  add_tensor_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_27: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_75, add_26);  view_75 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_15: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_27, arg28_1);  add_27 = arg28_1 = None
    add_28: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_15, arg29_1);  mul_15 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_76: "f32[128, 128]" = torch.ops.aten.reshape.default(add_28, [128, 128])
    permute_38: "f32[128, 512]" = torch.ops.aten.permute.default(arg448_1, [1, 0]);  arg448_1 = None
    
    # No stacktrace found for following nodes
    mm_default_333: "f32[128, 512]" = torch.ops.aten.mm.default(view_76, permute_38);  view_76 = permute_38 = None
    add_tensor_333: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_333, arg449_1);  mm_default_333 = arg449_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_77: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_333, [1, 128, 512]);  add_tensor_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_7: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_77);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_78: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_7, [128, 512]);  relu_7 = None
    permute_39: "f32[512, 128]" = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
    
    # No stacktrace found for following nodes
    mm_default_332: "f32[128, 128]" = torch.ops.aten.mm.default(view_78, permute_39);  view_78 = permute_39 = None
    add_tensor_332: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_332, arg451_1);  mm_default_332 = arg451_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_79: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_332, [1, 128, 128]);  add_tensor_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_29: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_79, add_28);  view_79 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_16: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_29, arg30_1);  add_29 = arg30_1 = None
    add_30: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_16, arg31_1);  mul_16 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_80: "f32[128, 128]" = torch.ops.aten.reshape.default(add_30, [128, 128]);  add_30 = None
    permute_40: "f32[128, 512]" = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
    
    # No stacktrace found for following nodes
    mm_default_331: "f32[128, 512]" = torch.ops.aten.mm.default(view_80, permute_40);  view_80 = permute_40 = None
    add_tensor_331: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_331, arg453_1);  mm_default_331 = arg453_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_81: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_331, [1, 128, 512]);  add_tensor_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_31: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_81, add_17);  view_81 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_17: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_31, arg32_1);  add_31 = arg32_1 = None
    add_32: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_17, arg33_1);  mul_17 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_84: "f32[128, 512]" = torch.ops.aten.reshape.default(add_32, [128, 512])
    permute_42: "f32[512, 128]" = torch.ops.aten.permute.default(arg456_1, [1, 0]);  arg456_1 = None
    
    # No stacktrace found for following nodes
    mm_default_330: "f32[128, 128]" = torch.ops.aten.mm.default(view_84, permute_42);  view_84 = permute_42 = None
    add_tensor_330: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_330, arg457_1);  mm_default_330 = arg457_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_85: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_330, [1, 128, 128]);  add_tensor_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_19: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_85, arg36_1);  view_85 = arg36_1 = None
    add_34: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, arg37_1);  mul_19 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_86: "f32[128, 128]" = torch.ops.aten.reshape.default(add_34, [128, 128])
    permute_43: "f32[128, 128]" = torch.ops.aten.permute.default(arg458_1, [1, 0]);  arg458_1 = None
    
    # No stacktrace found for following nodes
    mm_default_329: "f32[128, 128]" = torch.ops.aten.mm.default(view_86, permute_43);  view_86 = permute_43 = None
    add_tensor_329: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_329, arg459_1);  mm_default_329 = arg459_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_87: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_329, [1, 128, 128]);  add_tensor_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_92: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_87, [1, 128, 4, 32]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # No stacktrace found for following nodes
    clone_default_63: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_88: "f32[128, 128]" = torch.ops.aten.reshape.default(add_34, [128, 128]);  add_34 = None
    permute_44: "f32[128, 128]" = torch.ops.aten.permute.default(arg460_1, [1, 0]);  arg460_1 = None
    
    # No stacktrace found for following nodes
    mm_default_328: "f32[128, 128]" = torch.ops.aten.mm.default(view_88, permute_44);  view_88 = permute_44 = None
    add_tensor_328: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_328, arg461_1);  mm_default_328 = arg461_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_89: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_328, [1, 128, 128]);  add_tensor_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_93: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_89, [1, 128, 4, 32]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # No stacktrace found for following nodes
    clone_default_64: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_90: "f32[128, 512]" = torch.ops.aten.reshape.default(add_32, [128, 512])
    permute_45: "f32[512, 128]" = torch.ops.aten.permute.default(arg462_1, [1, 0]);  arg462_1 = None
    
    # No stacktrace found for following nodes
    mm_default_327: "f32[128, 128]" = torch.ops.aten.mm.default(view_90, permute_45);  view_90 = permute_45 = None
    add_tensor_327: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_327, arg463_1);  mm_default_327 = arg463_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_91: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_327, [1, 128, 128]);  add_tensor_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_94: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_91, [1, 128, 4, 32]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # No stacktrace found for following nodes
    clone_default_65: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_63, clone_default_64, clone_default_65, None, False, scale = 0.17677669529663687);  clone_default_63 = clone_default_64 = clone_default_65 = None
    getitem_23: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_21[0];  _scaled_dot_product_efficient_attention_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_50: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_23, [0, 2, 1, 3]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_101: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_50, [1, 128, 128]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_102: "f32[128, 128]" = torch.ops.aten.reshape.default(view_101, [128, 128]);  view_101 = None
    permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(arg464_1, [1, 0]);  arg464_1 = None
    
    # No stacktrace found for following nodes
    mm_default_326: "f32[128, 128]" = torch.ops.aten.mm.default(view_102, permute_51);  view_102 = permute_51 = None
    add_tensor_326: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_326, arg465_1);  mm_default_326 = arg465_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_103: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_326, [1, 128, 128]);  add_tensor_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_82: "f32[128, 512]" = torch.ops.aten.reshape.default(add_32, [128, 512])
    permute_41: "f32[512, 128]" = torch.ops.aten.permute.default(arg454_1, [1, 0]);  arg454_1 = None
    
    # No stacktrace found for following nodes
    mm_default_325: "f32[128, 128]" = torch.ops.aten.mm.default(view_82, permute_41);  view_82 = permute_41 = None
    add_tensor_325: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_325, arg455_1);  mm_default_325 = arg455_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_83: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_325, [1, 128, 128]);  add_tensor_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_18: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_83, arg34_1);  view_83 = arg34_1 = None
    add_33: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_18, arg35_1);  mul_18 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_36: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_103, add_33);  view_103 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_20: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_36, arg38_1);  add_36 = arg38_1 = None
    add_37: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, arg39_1);  mul_20 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[128, 128]" = torch.ops.aten.reshape.default(add_37, [128, 128])
    permute_52: "f32[128, 512]" = torch.ops.aten.permute.default(arg466_1, [1, 0]);  arg466_1 = None
    
    # No stacktrace found for following nodes
    mm_default_324: "f32[128, 512]" = torch.ops.aten.mm.default(view_104, permute_52);  view_104 = permute_52 = None
    add_tensor_324: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_324, arg467_1);  mm_default_324 = arg467_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_105: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_324, [1, 128, 512]);  add_tensor_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_8: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_106: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_8, [128, 512]);  relu_8 = None
    permute_53: "f32[512, 128]" = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
    
    # No stacktrace found for following nodes
    mm_default_323: "f32[128, 128]" = torch.ops.aten.mm.default(view_106, permute_53);  view_106 = permute_53 = None
    add_tensor_323: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_323, arg469_1);  mm_default_323 = arg469_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_107: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_323, [1, 128, 128]);  add_tensor_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_38: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_107, add_37);  view_107 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_21: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_38, arg40_1);  add_38 = arg40_1 = None
    add_39: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, arg41_1);  mul_21 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[128, 128]" = torch.ops.aten.reshape.default(add_39, [128, 128])
    permute_54: "f32[128, 512]" = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
    
    # No stacktrace found for following nodes
    mm_default_322: "f32[128, 512]" = torch.ops.aten.mm.default(view_108, permute_54);  view_108 = permute_54 = None
    add_tensor_322: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_322, arg471_1);  mm_default_322 = arg471_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_109: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_322, [1, 128, 512]);  add_tensor_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_9: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_110: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_9, [128, 512]);  relu_9 = None
    permute_55: "f32[512, 128]" = torch.ops.aten.permute.default(arg472_1, [1, 0]);  arg472_1 = None
    
    # No stacktrace found for following nodes
    mm_default_321: "f32[128, 128]" = torch.ops.aten.mm.default(view_110, permute_55);  view_110 = permute_55 = None
    add_tensor_321: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_321, arg473_1);  mm_default_321 = arg473_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_111: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_321, [1, 128, 128]);  add_tensor_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_40: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_111, add_39);  view_111 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_22: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_40, arg42_1);  add_40 = arg42_1 = None
    add_41: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, arg43_1);  mul_22 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_112: "f32[128, 128]" = torch.ops.aten.reshape.default(add_41, [128, 128])
    permute_56: "f32[128, 512]" = torch.ops.aten.permute.default(arg474_1, [1, 0]);  arg474_1 = None
    
    # No stacktrace found for following nodes
    mm_default_320: "f32[128, 512]" = torch.ops.aten.mm.default(view_112, permute_56);  view_112 = permute_56 = None
    add_tensor_320: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_320, arg475_1);  mm_default_320 = arg475_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_113: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_320, [1, 128, 512]);  add_tensor_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_10: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_114: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_10, [128, 512]);  relu_10 = None
    permute_57: "f32[512, 128]" = torch.ops.aten.permute.default(arg476_1, [1, 0]);  arg476_1 = None
    
    # No stacktrace found for following nodes
    mm_default_319: "f32[128, 128]" = torch.ops.aten.mm.default(view_114, permute_57);  view_114 = permute_57 = None
    add_tensor_319: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_319, arg477_1);  mm_default_319 = arg477_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_115: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_319, [1, 128, 128]);  add_tensor_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_42: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_115, add_41);  view_115 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_23: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_42, arg44_1);  add_42 = arg44_1 = None
    add_43: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_23, arg45_1);  mul_23 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_116: "f32[128, 128]" = torch.ops.aten.reshape.default(add_43, [128, 128])
    permute_58: "f32[128, 512]" = torch.ops.aten.permute.default(arg478_1, [1, 0]);  arg478_1 = None
    
    # No stacktrace found for following nodes
    mm_default_318: "f32[128, 512]" = torch.ops.aten.mm.default(view_116, permute_58);  view_116 = permute_58 = None
    add_tensor_318: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_318, arg479_1);  mm_default_318 = arg479_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_117: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_318, [1, 128, 512]);  add_tensor_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_11: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_117);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_118: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_11, [128, 512]);  relu_11 = None
    permute_59: "f32[512, 128]" = torch.ops.aten.permute.default(arg480_1, [1, 0]);  arg480_1 = None
    
    # No stacktrace found for following nodes
    mm_default_317: "f32[128, 128]" = torch.ops.aten.mm.default(view_118, permute_59);  view_118 = permute_59 = None
    add_tensor_317: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_317, arg481_1);  mm_default_317 = arg481_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_119: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_317, [1, 128, 128]);  add_tensor_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_44: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_119, add_43);  view_119 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_24: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_44, arg46_1);  add_44 = arg46_1 = None
    add_45: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_24, arg47_1);  mul_24 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_120: "f32[128, 128]" = torch.ops.aten.reshape.default(add_45, [128, 128]);  add_45 = None
    permute_60: "f32[128, 512]" = torch.ops.aten.permute.default(arg482_1, [1, 0]);  arg482_1 = None
    
    # No stacktrace found for following nodes
    mm_default_316: "f32[128, 512]" = torch.ops.aten.mm.default(view_120, permute_60);  view_120 = permute_60 = None
    add_tensor_316: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_316, arg483_1);  mm_default_316 = arg483_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_121: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_316, [1, 128, 512]);  add_tensor_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_46: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_121, add_32);  view_121 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_25: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_46, arg48_1);  add_46 = arg48_1 = None
    add_47: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_25, arg49_1);  mul_25 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_124: "f32[128, 512]" = torch.ops.aten.reshape.default(add_47, [128, 512])
    permute_62: "f32[512, 128]" = torch.ops.aten.permute.default(arg486_1, [1, 0]);  arg486_1 = None
    
    # No stacktrace found for following nodes
    mm_default_315: "f32[128, 128]" = torch.ops.aten.mm.default(view_124, permute_62);  view_124 = permute_62 = None
    add_tensor_315: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_315, arg487_1);  mm_default_315 = arg487_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_125: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_315, [1, 128, 128]);  add_tensor_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_27: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_125, arg52_1);  view_125 = arg52_1 = None
    add_49: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_27, arg53_1);  mul_27 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_126: "f32[128, 128]" = torch.ops.aten.reshape.default(add_49, [128, 128])
    permute_63: "f32[128, 128]" = torch.ops.aten.permute.default(arg488_1, [1, 0]);  arg488_1 = None
    
    # No stacktrace found for following nodes
    mm_default_314: "f32[128, 128]" = torch.ops.aten.mm.default(view_126, permute_63);  view_126 = permute_63 = None
    add_tensor_314: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_314, arg489_1);  mm_default_314 = arg489_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_127: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_314, [1, 128, 128]);  add_tensor_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_132: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_127, [1, 128, 4, 32]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_66: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # No stacktrace found for following nodes
    clone_default_60: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_128: "f32[128, 128]" = torch.ops.aten.reshape.default(add_49, [128, 128]);  add_49 = None
    permute_64: "f32[128, 128]" = torch.ops.aten.permute.default(arg490_1, [1, 0]);  arg490_1 = None
    
    # No stacktrace found for following nodes
    mm_default_313: "f32[128, 128]" = torch.ops.aten.mm.default(view_128, permute_64);  view_128 = permute_64 = None
    add_tensor_313: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_313, arg491_1);  mm_default_313 = arg491_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_129: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_313, [1, 128, 128]);  add_tensor_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_133: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_129, [1, 128, 4, 32]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_67: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # No stacktrace found for following nodes
    clone_default_61: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_130: "f32[128, 512]" = torch.ops.aten.reshape.default(add_47, [128, 512])
    permute_65: "f32[512, 128]" = torch.ops.aten.permute.default(arg492_1, [1, 0]);  arg492_1 = None
    
    # No stacktrace found for following nodes
    mm_default_312: "f32[128, 128]" = torch.ops.aten.mm.default(view_130, permute_65);  view_130 = permute_65 = None
    add_tensor_312: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_312, arg493_1);  mm_default_312 = arg493_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_131: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_312, [1, 128, 128]);  add_tensor_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_134: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_131, [1, 128, 4, 32]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    
    # No stacktrace found for following nodes
    clone_default_62: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_60, clone_default_61, clone_default_62, None, False, scale = 0.17677669529663687);  clone_default_60 = clone_default_61 = clone_default_62 = None
    getitem_22: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_20[0];  _scaled_dot_product_efficient_attention_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_70: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_22, [0, 2, 1, 3]);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_141: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_70, [1, 128, 128]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_142: "f32[128, 128]" = torch.ops.aten.reshape.default(view_141, [128, 128]);  view_141 = None
    permute_71: "f32[128, 128]" = torch.ops.aten.permute.default(arg494_1, [1, 0]);  arg494_1 = None
    
    # No stacktrace found for following nodes
    mm_default_311: "f32[128, 128]" = torch.ops.aten.mm.default(view_142, permute_71);  view_142 = permute_71 = None
    add_tensor_311: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_311, arg495_1);  mm_default_311 = arg495_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_143: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_311, [1, 128, 128]);  add_tensor_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_122: "f32[128, 512]" = torch.ops.aten.reshape.default(add_47, [128, 512])
    permute_61: "f32[512, 128]" = torch.ops.aten.permute.default(arg484_1, [1, 0]);  arg484_1 = None
    
    # No stacktrace found for following nodes
    mm_default_310: "f32[128, 128]" = torch.ops.aten.mm.default(view_122, permute_61);  view_122 = permute_61 = None
    add_tensor_310: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_310, arg485_1);  mm_default_310 = arg485_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_123: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_310, [1, 128, 128]);  add_tensor_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_26: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_123, arg50_1);  view_123 = arg50_1 = None
    add_48: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, arg51_1);  mul_26 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_51: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_143, add_48);  view_143 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_28: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_51, arg54_1);  add_51 = arg54_1 = None
    add_52: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_28, arg55_1);  mul_28 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_144: "f32[128, 128]" = torch.ops.aten.reshape.default(add_52, [128, 128])
    permute_72: "f32[128, 512]" = torch.ops.aten.permute.default(arg496_1, [1, 0]);  arg496_1 = None
    
    # No stacktrace found for following nodes
    mm_default_309: "f32[128, 512]" = torch.ops.aten.mm.default(view_144, permute_72);  view_144 = permute_72 = None
    add_tensor_309: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_309, arg497_1);  mm_default_309 = arg497_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_145: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_309, [1, 128, 512]);  add_tensor_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_12: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_145);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_146: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_12, [128, 512]);  relu_12 = None
    permute_73: "f32[512, 128]" = torch.ops.aten.permute.default(arg498_1, [1, 0]);  arg498_1 = None
    
    # No stacktrace found for following nodes
    mm_default_308: "f32[128, 128]" = torch.ops.aten.mm.default(view_146, permute_73);  view_146 = permute_73 = None
    add_tensor_308: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_308, arg499_1);  mm_default_308 = arg499_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_147: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_308, [1, 128, 128]);  add_tensor_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_53: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_147, add_52);  view_147 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_29: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_53, arg56_1);  add_53 = arg56_1 = None
    add_54: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_29, arg57_1);  mul_29 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[128, 128]" = torch.ops.aten.reshape.default(add_54, [128, 128])
    permute_74: "f32[128, 512]" = torch.ops.aten.permute.default(arg500_1, [1, 0]);  arg500_1 = None
    
    # No stacktrace found for following nodes
    mm_default_307: "f32[128, 512]" = torch.ops.aten.mm.default(view_148, permute_74);  view_148 = permute_74 = None
    add_tensor_307: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_307, arg501_1);  mm_default_307 = arg501_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_149: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_307, [1, 128, 512]);  add_tensor_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_13: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_149);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_150: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_13, [128, 512]);  relu_13 = None
    permute_75: "f32[512, 128]" = torch.ops.aten.permute.default(arg502_1, [1, 0]);  arg502_1 = None
    
    # No stacktrace found for following nodes
    mm_default_306: "f32[128, 128]" = torch.ops.aten.mm.default(view_150, permute_75);  view_150 = permute_75 = None
    add_tensor_306: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_306, arg503_1);  mm_default_306 = arg503_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_151: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_306, [1, 128, 128]);  add_tensor_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_55: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_151, add_54);  view_151 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_30: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_55, arg58_1);  add_55 = arg58_1 = None
    add_56: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, arg59_1);  mul_30 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[128, 128]" = torch.ops.aten.reshape.default(add_56, [128, 128])
    permute_76: "f32[128, 512]" = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
    
    # No stacktrace found for following nodes
    mm_default_305: "f32[128, 512]" = torch.ops.aten.mm.default(view_152, permute_76);  view_152 = permute_76 = None
    add_tensor_305: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_305, arg505_1);  mm_default_305 = arg505_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_153: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_305, [1, 128, 512]);  add_tensor_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_14: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_154: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_14, [128, 512]);  relu_14 = None
    permute_77: "f32[512, 128]" = torch.ops.aten.permute.default(arg506_1, [1, 0]);  arg506_1 = None
    
    # No stacktrace found for following nodes
    mm_default_304: "f32[128, 128]" = torch.ops.aten.mm.default(view_154, permute_77);  view_154 = permute_77 = None
    add_tensor_304: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_304, arg507_1);  mm_default_304 = arg507_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_155: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_304, [1, 128, 128]);  add_tensor_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_57: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_155, add_56);  view_155 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_31: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_57, arg60_1);  add_57 = arg60_1 = None
    add_58: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_31, arg61_1);  mul_31 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_156: "f32[128, 128]" = torch.ops.aten.reshape.default(add_58, [128, 128])
    permute_78: "f32[128, 512]" = torch.ops.aten.permute.default(arg508_1, [1, 0]);  arg508_1 = None
    
    # No stacktrace found for following nodes
    mm_default_303: "f32[128, 512]" = torch.ops.aten.mm.default(view_156, permute_78);  view_156 = permute_78 = None
    add_tensor_303: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_303, arg509_1);  mm_default_303 = arg509_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_157: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_303, [1, 128, 512]);  add_tensor_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_15: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_157);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_158: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_15, [128, 512]);  relu_15 = None
    permute_79: "f32[512, 128]" = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
    
    # No stacktrace found for following nodes
    mm_default_302: "f32[128, 128]" = torch.ops.aten.mm.default(view_158, permute_79);  view_158 = permute_79 = None
    add_tensor_302: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_302, arg511_1);  mm_default_302 = arg511_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_159: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_302, [1, 128, 128]);  add_tensor_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_59: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_159, add_58);  view_159 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_32: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_59, arg62_1);  add_59 = arg62_1 = None
    add_60: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_32, arg63_1);  mul_32 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_160: "f32[128, 128]" = torch.ops.aten.reshape.default(add_60, [128, 128]);  add_60 = None
    permute_80: "f32[128, 512]" = torch.ops.aten.permute.default(arg512_1, [1, 0]);  arg512_1 = None
    
    # No stacktrace found for following nodes
    mm_default_301: "f32[128, 512]" = torch.ops.aten.mm.default(view_160, permute_80);  view_160 = permute_80 = None
    add_tensor_301: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_301, arg513_1);  mm_default_301 = arg513_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_161: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_301, [1, 128, 512]);  add_tensor_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_61: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_161, add_47);  view_161 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_61, arg64_1);  add_61 = arg64_1 = None
    add_62: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_33, arg65_1);  mul_33 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_164: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_82: "f32[512, 128]" = torch.ops.aten.permute.default(arg516_1, [1, 0]);  arg516_1 = None
    
    # No stacktrace found for following nodes
    mm_default_300: "f32[128, 128]" = torch.ops.aten.mm.default(view_164, permute_82);  view_164 = permute_82 = None
    add_tensor_300: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_300, arg517_1);  mm_default_300 = arg517_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_165: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_300, [1, 128, 128]);  add_tensor_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_35: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_165, arg68_1);  view_165 = arg68_1 = None
    add_64: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_166: "f32[128, 128]" = torch.ops.aten.reshape.default(add_64, [128, 128])
    permute_83: "f32[128, 128]" = torch.ops.aten.permute.default(arg518_1, [1, 0]);  arg518_1 = None
    
    # No stacktrace found for following nodes
    mm_default_299: "f32[128, 128]" = torch.ops.aten.mm.default(view_166, permute_83);  view_166 = permute_83 = None
    add_tensor_299: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_299, arg519_1);  mm_default_299 = arg519_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_167: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_299, [1, 128, 128]);  add_tensor_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_172: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_167, [1, 128, 4, 32]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_86: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # No stacktrace found for following nodes
    clone_default_57: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_168: "f32[128, 128]" = torch.ops.aten.reshape.default(add_64, [128, 128]);  add_64 = None
    permute_84: "f32[128, 128]" = torch.ops.aten.permute.default(arg520_1, [1, 0]);  arg520_1 = None
    
    # No stacktrace found for following nodes
    mm_default_298: "f32[128, 128]" = torch.ops.aten.mm.default(view_168, permute_84);  view_168 = permute_84 = None
    add_tensor_298: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_298, arg521_1);  mm_default_298 = arg521_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_169: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_298, [1, 128, 128]);  add_tensor_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_173: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_169, [1, 128, 4, 32]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_87: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    
    # No stacktrace found for following nodes
    clone_default_58: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_170: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_85: "f32[512, 128]" = torch.ops.aten.permute.default(arg522_1, [1, 0]);  arg522_1 = None
    
    # No stacktrace found for following nodes
    mm_default_297: "f32[128, 128]" = torch.ops.aten.mm.default(view_170, permute_85);  view_170 = permute_85 = None
    add_tensor_297: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_297, arg523_1);  mm_default_297 = arg523_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_171: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_297, [1, 128, 128]);  add_tensor_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_174: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_171, [1, 128, 4, 32]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_88: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # No stacktrace found for following nodes
    clone_default_59: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_57, clone_default_58, clone_default_59, None, False, scale = 0.17677669529663687);  clone_default_57 = clone_default_58 = clone_default_59 = None
    getitem_21: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_19[0];  _scaled_dot_product_efficient_attention_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_90: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_181: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_90, [1, 128, 128]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_182: "f32[128, 128]" = torch.ops.aten.reshape.default(view_181, [128, 128]);  view_181 = None
    permute_91: "f32[128, 128]" = torch.ops.aten.permute.default(arg524_1, [1, 0]);  arg524_1 = None
    
    # No stacktrace found for following nodes
    mm_default_296: "f32[128, 128]" = torch.ops.aten.mm.default(view_182, permute_91);  view_182 = permute_91 = None
    add_tensor_296: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_296, arg525_1);  mm_default_296 = arg525_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_183: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_296, [1, 128, 128]);  add_tensor_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_162: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_81: "f32[512, 128]" = torch.ops.aten.permute.default(arg514_1, [1, 0]);  arg514_1 = None
    
    # No stacktrace found for following nodes
    mm_default_295: "f32[128, 128]" = torch.ops.aten.mm.default(view_162, permute_81);  view_162 = permute_81 = None
    add_tensor_295: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_295, arg515_1);  mm_default_295 = arg515_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_163: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_295, [1, 128, 128]);  add_tensor_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_34: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_163, arg66_1);  view_163 = arg66_1 = None
    add_63: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_34, arg67_1);  mul_34 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_66: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_183, add_63);  view_183 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_36: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_66, arg70_1);  add_66 = arg70_1 = None
    add_67: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_36, arg71_1);  mul_36 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_184: "f32[128, 128]" = torch.ops.aten.reshape.default(add_67, [128, 128])
    permute_92: "f32[128, 512]" = torch.ops.aten.permute.default(arg526_1, [1, 0]);  arg526_1 = None
    
    # No stacktrace found for following nodes
    mm_default_294: "f32[128, 512]" = torch.ops.aten.mm.default(view_184, permute_92);  view_184 = permute_92 = None
    add_tensor_294: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_294, arg527_1);  mm_default_294 = arg527_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_185: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_294, [1, 128, 512]);  add_tensor_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_16: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_185);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_186: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_16, [128, 512]);  relu_16 = None
    permute_93: "f32[512, 128]" = torch.ops.aten.permute.default(arg528_1, [1, 0]);  arg528_1 = None
    
    # No stacktrace found for following nodes
    mm_default_293: "f32[128, 128]" = torch.ops.aten.mm.default(view_186, permute_93);  view_186 = permute_93 = None
    add_tensor_293: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_293, arg529_1);  mm_default_293 = arg529_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_187: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_293, [1, 128, 128]);  add_tensor_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_68: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_187, add_67);  view_187 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_37: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_68, arg72_1);  add_68 = arg72_1 = None
    add_69: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, arg73_1);  mul_37 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_188: "f32[128, 128]" = torch.ops.aten.reshape.default(add_69, [128, 128])
    permute_94: "f32[128, 512]" = torch.ops.aten.permute.default(arg530_1, [1, 0]);  arg530_1 = None
    
    # No stacktrace found for following nodes
    mm_default_292: "f32[128, 512]" = torch.ops.aten.mm.default(view_188, permute_94);  view_188 = permute_94 = None
    add_tensor_292: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_292, arg531_1);  mm_default_292 = arg531_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_189: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_292, [1, 128, 512]);  add_tensor_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_17: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_190: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_17, [128, 512]);  relu_17 = None
    permute_95: "f32[512, 128]" = torch.ops.aten.permute.default(arg532_1, [1, 0]);  arg532_1 = None
    
    # No stacktrace found for following nodes
    mm_default_291: "f32[128, 128]" = torch.ops.aten.mm.default(view_190, permute_95);  view_190 = permute_95 = None
    add_tensor_291: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_291, arg533_1);  mm_default_291 = arg533_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_191: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_291, [1, 128, 128]);  add_tensor_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_70: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_191, add_69);  view_191 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_38: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_70, arg74_1);  add_70 = arg74_1 = None
    add_71: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, arg75_1);  mul_38 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[128, 128]" = torch.ops.aten.reshape.default(add_71, [128, 128])
    permute_96: "f32[128, 512]" = torch.ops.aten.permute.default(arg534_1, [1, 0]);  arg534_1 = None
    
    # No stacktrace found for following nodes
    mm_default_290: "f32[128, 512]" = torch.ops.aten.mm.default(view_192, permute_96);  view_192 = permute_96 = None
    add_tensor_290: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_290, arg535_1);  mm_default_290 = arg535_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_193: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_290, [1, 128, 512]);  add_tensor_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_18: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_193);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_194: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_18, [128, 512]);  relu_18 = None
    permute_97: "f32[512, 128]" = torch.ops.aten.permute.default(arg536_1, [1, 0]);  arg536_1 = None
    
    # No stacktrace found for following nodes
    mm_default_289: "f32[128, 128]" = torch.ops.aten.mm.default(view_194, permute_97);  view_194 = permute_97 = None
    add_tensor_289: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_289, arg537_1);  mm_default_289 = arg537_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_195: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_289, [1, 128, 128]);  add_tensor_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_72: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_195, add_71);  view_195 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_39: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_72, arg76_1);  add_72 = arg76_1 = None
    add_73: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_39, arg77_1);  mul_39 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[128, 128]" = torch.ops.aten.reshape.default(add_73, [128, 128])
    permute_98: "f32[128, 512]" = torch.ops.aten.permute.default(arg538_1, [1, 0]);  arg538_1 = None
    
    # No stacktrace found for following nodes
    mm_default_288: "f32[128, 512]" = torch.ops.aten.mm.default(view_196, permute_98);  view_196 = permute_98 = None
    add_tensor_288: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_288, arg539_1);  mm_default_288 = arg539_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_197: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_288, [1, 128, 512]);  add_tensor_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_19: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_197);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_198: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_19, [128, 512]);  relu_19 = None
    permute_99: "f32[512, 128]" = torch.ops.aten.permute.default(arg540_1, [1, 0]);  arg540_1 = None
    
    # No stacktrace found for following nodes
    mm_default_287: "f32[128, 128]" = torch.ops.aten.mm.default(view_198, permute_99);  view_198 = permute_99 = None
    add_tensor_287: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_287, arg541_1);  mm_default_287 = arg541_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_199: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_287, [1, 128, 128]);  add_tensor_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_74: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_199, add_73);  view_199 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_40: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_74, arg78_1);  add_74 = arg78_1 = None
    add_75: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_40, arg79_1);  mul_40 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_200: "f32[128, 128]" = torch.ops.aten.reshape.default(add_75, [128, 128]);  add_75 = None
    permute_100: "f32[128, 512]" = torch.ops.aten.permute.default(arg542_1, [1, 0]);  arg542_1 = None
    
    # No stacktrace found for following nodes
    mm_default_286: "f32[128, 512]" = torch.ops.aten.mm.default(view_200, permute_100);  view_200 = permute_100 = None
    add_tensor_286: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_286, arg543_1);  mm_default_286 = arg543_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_201: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_286, [1, 128, 512]);  add_tensor_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_76: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_201, add_62);  view_201 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_76, arg80_1);  add_76 = arg80_1 = None
    add_77: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_41, arg81_1);  mul_41 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_204: "f32[128, 512]" = torch.ops.aten.reshape.default(add_77, [128, 512])
    permute_102: "f32[512, 128]" = torch.ops.aten.permute.default(arg546_1, [1, 0]);  arg546_1 = None
    
    # No stacktrace found for following nodes
    mm_default_285: "f32[128, 128]" = torch.ops.aten.mm.default(view_204, permute_102);  view_204 = permute_102 = None
    add_tensor_285: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_285, arg547_1);  mm_default_285 = arg547_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_205: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_285, [1, 128, 128]);  add_tensor_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_43: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_205, arg84_1);  view_205 = arg84_1 = None
    add_79: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_43, arg85_1);  mul_43 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_206: "f32[128, 128]" = torch.ops.aten.reshape.default(add_79, [128, 128])
    permute_103: "f32[128, 128]" = torch.ops.aten.permute.default(arg548_1, [1, 0]);  arg548_1 = None
    
    # No stacktrace found for following nodes
    mm_default_284: "f32[128, 128]" = torch.ops.aten.mm.default(view_206, permute_103);  view_206 = permute_103 = None
    add_tensor_284: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_284, arg549_1);  mm_default_284 = arg549_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_207: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_284, [1, 128, 128]);  add_tensor_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_212: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_207, [1, 128, 4, 32]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_106: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # No stacktrace found for following nodes
    clone_default_54: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_208: "f32[128, 128]" = torch.ops.aten.reshape.default(add_79, [128, 128]);  add_79 = None
    permute_104: "f32[128, 128]" = torch.ops.aten.permute.default(arg550_1, [1, 0]);  arg550_1 = None
    
    # No stacktrace found for following nodes
    mm_default_283: "f32[128, 128]" = torch.ops.aten.mm.default(view_208, permute_104);  view_208 = permute_104 = None
    add_tensor_283: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_283, arg551_1);  mm_default_283 = arg551_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_209: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_283, [1, 128, 128]);  add_tensor_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_213: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_209, [1, 128, 4, 32]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_107: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # No stacktrace found for following nodes
    clone_default_55: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_210: "f32[128, 512]" = torch.ops.aten.reshape.default(add_77, [128, 512])
    permute_105: "f32[512, 128]" = torch.ops.aten.permute.default(arg552_1, [1, 0]);  arg552_1 = None
    
    # No stacktrace found for following nodes
    mm_default_282: "f32[128, 128]" = torch.ops.aten.mm.default(view_210, permute_105);  view_210 = permute_105 = None
    add_tensor_282: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_282, arg553_1);  mm_default_282 = arg553_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_211: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_282, [1, 128, 128]);  add_tensor_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_214: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_211, [1, 128, 4, 32]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_108: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    
    # No stacktrace found for following nodes
    clone_default_56: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_54, clone_default_55, clone_default_56, None, False, scale = 0.17677669529663687);  clone_default_54 = clone_default_55 = clone_default_56 = None
    getitem_20: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_18[0];  _scaled_dot_product_efficient_attention_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_110: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_221: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_110, [1, 128, 128]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_222: "f32[128, 128]" = torch.ops.aten.reshape.default(view_221, [128, 128]);  view_221 = None
    permute_111: "f32[128, 128]" = torch.ops.aten.permute.default(arg554_1, [1, 0]);  arg554_1 = None
    
    # No stacktrace found for following nodes
    mm_default_281: "f32[128, 128]" = torch.ops.aten.mm.default(view_222, permute_111);  view_222 = permute_111 = None
    add_tensor_281: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_281, arg555_1);  mm_default_281 = arg555_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_223: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_281, [1, 128, 128]);  add_tensor_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_202: "f32[128, 512]" = torch.ops.aten.reshape.default(add_77, [128, 512])
    permute_101: "f32[512, 128]" = torch.ops.aten.permute.default(arg544_1, [1, 0]);  arg544_1 = None
    
    # No stacktrace found for following nodes
    mm_default_280: "f32[128, 128]" = torch.ops.aten.mm.default(view_202, permute_101);  view_202 = permute_101 = None
    add_tensor_280: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_280, arg545_1);  mm_default_280 = arg545_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_203: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_280, [1, 128, 128]);  add_tensor_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_42: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_203, arg82_1);  view_203 = arg82_1 = None
    add_78: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_42, arg83_1);  mul_42 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_81: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_223, add_78);  view_223 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_44: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_81, arg86_1);  add_81 = arg86_1 = None
    add_82: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_44, arg87_1);  mul_44 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_224: "f32[128, 128]" = torch.ops.aten.reshape.default(add_82, [128, 128])
    permute_112: "f32[128, 512]" = torch.ops.aten.permute.default(arg556_1, [1, 0]);  arg556_1 = None
    
    # No stacktrace found for following nodes
    mm_default_279: "f32[128, 512]" = torch.ops.aten.mm.default(view_224, permute_112);  view_224 = permute_112 = None
    add_tensor_279: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_279, arg557_1);  mm_default_279 = arg557_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_225: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_279, [1, 128, 512]);  add_tensor_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_20: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_225);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_226: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_20, [128, 512]);  relu_20 = None
    permute_113: "f32[512, 128]" = torch.ops.aten.permute.default(arg558_1, [1, 0]);  arg558_1 = None
    
    # No stacktrace found for following nodes
    mm_default_278: "f32[128, 128]" = torch.ops.aten.mm.default(view_226, permute_113);  view_226 = permute_113 = None
    add_tensor_278: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_278, arg559_1);  mm_default_278 = arg559_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_227: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_278, [1, 128, 128]);  add_tensor_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_83: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_227, add_82);  view_227 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_45: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_83, arg88_1);  add_83 = arg88_1 = None
    add_84: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_45, arg89_1);  mul_45 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_228: "f32[128, 128]" = torch.ops.aten.reshape.default(add_84, [128, 128])
    permute_114: "f32[128, 512]" = torch.ops.aten.permute.default(arg560_1, [1, 0]);  arg560_1 = None
    
    # No stacktrace found for following nodes
    mm_default_277: "f32[128, 512]" = torch.ops.aten.mm.default(view_228, permute_114);  view_228 = permute_114 = None
    add_tensor_277: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_277, arg561_1);  mm_default_277 = arg561_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_229: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_277, [1, 128, 512]);  add_tensor_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_21: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_229);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_230: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_21, [128, 512]);  relu_21 = None
    permute_115: "f32[512, 128]" = torch.ops.aten.permute.default(arg562_1, [1, 0]);  arg562_1 = None
    
    # No stacktrace found for following nodes
    mm_default_276: "f32[128, 128]" = torch.ops.aten.mm.default(view_230, permute_115);  view_230 = permute_115 = None
    add_tensor_276: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_276, arg563_1);  mm_default_276 = arg563_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_231: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_276, [1, 128, 128]);  add_tensor_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_85: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_231, add_84);  view_231 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_46: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_85, arg90_1);  add_85 = arg90_1 = None
    add_86: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, arg91_1);  mul_46 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_232: "f32[128, 128]" = torch.ops.aten.reshape.default(add_86, [128, 128])
    permute_116: "f32[128, 512]" = torch.ops.aten.permute.default(arg564_1, [1, 0]);  arg564_1 = None
    
    # No stacktrace found for following nodes
    mm_default_275: "f32[128, 512]" = torch.ops.aten.mm.default(view_232, permute_116);  view_232 = permute_116 = None
    add_tensor_275: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_275, arg565_1);  mm_default_275 = arg565_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_233: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_275, [1, 128, 512]);  add_tensor_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_22: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_233);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_234: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_22, [128, 512]);  relu_22 = None
    permute_117: "f32[512, 128]" = torch.ops.aten.permute.default(arg566_1, [1, 0]);  arg566_1 = None
    
    # No stacktrace found for following nodes
    mm_default_274: "f32[128, 128]" = torch.ops.aten.mm.default(view_234, permute_117);  view_234 = permute_117 = None
    add_tensor_274: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_274, arg567_1);  mm_default_274 = arg567_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_235: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_274, [1, 128, 128]);  add_tensor_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_87: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_235, add_86);  view_235 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_47: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_87, arg92_1);  add_87 = arg92_1 = None
    add_88: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_47, arg93_1);  mul_47 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[128, 128]" = torch.ops.aten.reshape.default(add_88, [128, 128])
    permute_118: "f32[128, 512]" = torch.ops.aten.permute.default(arg568_1, [1, 0]);  arg568_1 = None
    
    # No stacktrace found for following nodes
    mm_default_273: "f32[128, 512]" = torch.ops.aten.mm.default(view_236, permute_118);  view_236 = permute_118 = None
    add_tensor_273: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_273, arg569_1);  mm_default_273 = arg569_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_237: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_273, [1, 128, 512]);  add_tensor_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_23: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_237);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_238: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_23, [128, 512]);  relu_23 = None
    permute_119: "f32[512, 128]" = torch.ops.aten.permute.default(arg570_1, [1, 0]);  arg570_1 = None
    
    # No stacktrace found for following nodes
    mm_default_272: "f32[128, 128]" = torch.ops.aten.mm.default(view_238, permute_119);  view_238 = permute_119 = None
    add_tensor_272: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_272, arg571_1);  mm_default_272 = arg571_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_239: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_272, [1, 128, 128]);  add_tensor_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_89: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_239, add_88);  view_239 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_48: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_89, arg94_1);  add_89 = arg94_1 = None
    add_90: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_48, arg95_1);  mul_48 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_240: "f32[128, 128]" = torch.ops.aten.reshape.default(add_90, [128, 128]);  add_90 = None
    permute_120: "f32[128, 512]" = torch.ops.aten.permute.default(arg572_1, [1, 0]);  arg572_1 = None
    
    # No stacktrace found for following nodes
    mm_default_271: "f32[128, 512]" = torch.ops.aten.mm.default(view_240, permute_120);  view_240 = permute_120 = None
    add_tensor_271: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_271, arg573_1);  mm_default_271 = arg573_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_241: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_271, [1, 128, 512]);  add_tensor_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_91: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_241, add_77);  view_241 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_49: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_91, arg96_1);  add_91 = arg96_1 = None
    add_92: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_49, arg97_1);  mul_49 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_244: "f32[128, 512]" = torch.ops.aten.reshape.default(add_92, [128, 512])
    permute_122: "f32[512, 128]" = torch.ops.aten.permute.default(arg576_1, [1, 0]);  arg576_1 = None
    
    # No stacktrace found for following nodes
    mm_default_270: "f32[128, 128]" = torch.ops.aten.mm.default(view_244, permute_122);  view_244 = permute_122 = None
    add_tensor_270: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_270, arg577_1);  mm_default_270 = arg577_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_245: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_270, [1, 128, 128]);  add_tensor_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_51: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_245, arg100_1);  view_245 = arg100_1 = None
    add_94: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_51, arg101_1);  mul_51 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_246: "f32[128, 128]" = torch.ops.aten.reshape.default(add_94, [128, 128])
    permute_123: "f32[128, 128]" = torch.ops.aten.permute.default(arg578_1, [1, 0]);  arg578_1 = None
    
    # No stacktrace found for following nodes
    mm_default_269: "f32[128, 128]" = torch.ops.aten.mm.default(view_246, permute_123);  view_246 = permute_123 = None
    add_tensor_269: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_269, arg579_1);  mm_default_269 = arg579_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_247: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_269, [1, 128, 128]);  add_tensor_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_252: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_247, [1, 128, 4, 32]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # No stacktrace found for following nodes
    clone_default_51: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_248: "f32[128, 128]" = torch.ops.aten.reshape.default(add_94, [128, 128]);  add_94 = None
    permute_124: "f32[128, 128]" = torch.ops.aten.permute.default(arg580_1, [1, 0]);  arg580_1 = None
    
    # No stacktrace found for following nodes
    mm_default_268: "f32[128, 128]" = torch.ops.aten.mm.default(view_248, permute_124);  view_248 = permute_124 = None
    add_tensor_268: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_268, arg581_1);  mm_default_268 = arg581_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_249: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_268, [1, 128, 128]);  add_tensor_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_253: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_249, [1, 128, 4, 32]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    
    # No stacktrace found for following nodes
    clone_default_52: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_250: "f32[128, 512]" = torch.ops.aten.reshape.default(add_92, [128, 512])
    permute_125: "f32[512, 128]" = torch.ops.aten.permute.default(arg582_1, [1, 0]);  arg582_1 = None
    
    # No stacktrace found for following nodes
    mm_default_267: "f32[128, 128]" = torch.ops.aten.mm.default(view_250, permute_125);  view_250 = permute_125 = None
    add_tensor_267: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_267, arg583_1);  mm_default_267 = arg583_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_251: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_267, [1, 128, 128]);  add_tensor_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_254: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_251, [1, 128, 4, 32]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_128: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
    
    # No stacktrace found for following nodes
    clone_default_53: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_51, clone_default_52, clone_default_53, None, False, scale = 0.17677669529663687);  clone_default_51 = clone_default_52 = clone_default_53 = None
    getitem_19: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_17[0];  _scaled_dot_product_efficient_attention_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_130: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3]);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_261: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_130, [1, 128, 128]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_262: "f32[128, 128]" = torch.ops.aten.reshape.default(view_261, [128, 128]);  view_261 = None
    permute_131: "f32[128, 128]" = torch.ops.aten.permute.default(arg584_1, [1, 0]);  arg584_1 = None
    
    # No stacktrace found for following nodes
    mm_default_266: "f32[128, 128]" = torch.ops.aten.mm.default(view_262, permute_131);  view_262 = permute_131 = None
    add_tensor_266: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_266, arg585_1);  mm_default_266 = arg585_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_263: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_266, [1, 128, 128]);  add_tensor_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_242: "f32[128, 512]" = torch.ops.aten.reshape.default(add_92, [128, 512])
    permute_121: "f32[512, 128]" = torch.ops.aten.permute.default(arg574_1, [1, 0]);  arg574_1 = None
    
    # No stacktrace found for following nodes
    mm_default_265: "f32[128, 128]" = torch.ops.aten.mm.default(view_242, permute_121);  view_242 = permute_121 = None
    add_tensor_265: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_265, arg575_1);  mm_default_265 = arg575_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_243: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_265, [1, 128, 128]);  add_tensor_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_50: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_243, arg98_1);  view_243 = arg98_1 = None
    add_93: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_50, arg99_1);  mul_50 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_96: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_263, add_93);  view_263 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_52: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_96, arg102_1);  add_96 = arg102_1 = None
    add_97: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_52, arg103_1);  mul_52 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[128, 128]" = torch.ops.aten.reshape.default(add_97, [128, 128])
    permute_132: "f32[128, 512]" = torch.ops.aten.permute.default(arg586_1, [1, 0]);  arg586_1 = None
    
    # No stacktrace found for following nodes
    mm_default_264: "f32[128, 512]" = torch.ops.aten.mm.default(view_264, permute_132);  view_264 = permute_132 = None
    add_tensor_264: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_264, arg587_1);  mm_default_264 = arg587_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_265: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_264, [1, 128, 512]);  add_tensor_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_24: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_266: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_24, [128, 512]);  relu_24 = None
    permute_133: "f32[512, 128]" = torch.ops.aten.permute.default(arg588_1, [1, 0]);  arg588_1 = None
    
    # No stacktrace found for following nodes
    mm_default_263: "f32[128, 128]" = torch.ops.aten.mm.default(view_266, permute_133);  view_266 = permute_133 = None
    add_tensor_263: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_263, arg589_1);  mm_default_263 = arg589_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_267: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_263, [1, 128, 128]);  add_tensor_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_98: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_97);  view_267 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_53: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_98, arg104_1);  add_98 = arg104_1 = None
    add_99: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_53, arg105_1);  mul_53 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_268: "f32[128, 128]" = torch.ops.aten.reshape.default(add_99, [128, 128])
    permute_134: "f32[128, 512]" = torch.ops.aten.permute.default(arg590_1, [1, 0]);  arg590_1 = None
    
    # No stacktrace found for following nodes
    mm_default_262: "f32[128, 512]" = torch.ops.aten.mm.default(view_268, permute_134);  view_268 = permute_134 = None
    add_tensor_262: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_262, arg591_1);  mm_default_262 = arg591_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_269: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_262, [1, 128, 512]);  add_tensor_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_25: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_269);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_270: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_25, [128, 512]);  relu_25 = None
    permute_135: "f32[512, 128]" = torch.ops.aten.permute.default(arg592_1, [1, 0]);  arg592_1 = None
    
    # No stacktrace found for following nodes
    mm_default_261: "f32[128, 128]" = torch.ops.aten.mm.default(view_270, permute_135);  view_270 = permute_135 = None
    add_tensor_261: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_261, arg593_1);  mm_default_261 = arg593_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_271: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_261, [1, 128, 128]);  add_tensor_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_100: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_271, add_99);  view_271 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_54: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_100, arg106_1);  add_100 = arg106_1 = None
    add_101: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_54, arg107_1);  mul_54 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_272: "f32[128, 128]" = torch.ops.aten.reshape.default(add_101, [128, 128])
    permute_136: "f32[128, 512]" = torch.ops.aten.permute.default(arg594_1, [1, 0]);  arg594_1 = None
    
    # No stacktrace found for following nodes
    mm_default_260: "f32[128, 512]" = torch.ops.aten.mm.default(view_272, permute_136);  view_272 = permute_136 = None
    add_tensor_260: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_260, arg595_1);  mm_default_260 = arg595_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_273: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_260, [1, 128, 512]);  add_tensor_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_26: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_273);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_274: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_26, [128, 512]);  relu_26 = None
    permute_137: "f32[512, 128]" = torch.ops.aten.permute.default(arg596_1, [1, 0]);  arg596_1 = None
    
    # No stacktrace found for following nodes
    mm_default_259: "f32[128, 128]" = torch.ops.aten.mm.default(view_274, permute_137);  view_274 = permute_137 = None
    add_tensor_259: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_259, arg597_1);  mm_default_259 = arg597_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_275: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_259, [1, 128, 128]);  add_tensor_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_102: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_275, add_101);  view_275 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_55: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_102, arg108_1);  add_102 = arg108_1 = None
    add_103: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_55, arg109_1);  mul_55 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_276: "f32[128, 128]" = torch.ops.aten.reshape.default(add_103, [128, 128])
    permute_138: "f32[128, 512]" = torch.ops.aten.permute.default(arg598_1, [1, 0]);  arg598_1 = None
    
    # No stacktrace found for following nodes
    mm_default_258: "f32[128, 512]" = torch.ops.aten.mm.default(view_276, permute_138);  view_276 = permute_138 = None
    add_tensor_258: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_258, arg599_1);  mm_default_258 = arg599_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_277: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_258, [1, 128, 512]);  add_tensor_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_27: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_277);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_278: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_27, [128, 512]);  relu_27 = None
    permute_139: "f32[512, 128]" = torch.ops.aten.permute.default(arg600_1, [1, 0]);  arg600_1 = None
    
    # No stacktrace found for following nodes
    mm_default_257: "f32[128, 128]" = torch.ops.aten.mm.default(view_278, permute_139);  view_278 = permute_139 = None
    add_tensor_257: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_257, arg601_1);  mm_default_257 = arg601_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_279: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_257, [1, 128, 128]);  add_tensor_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_104: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_279, add_103);  view_279 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_56: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_104, arg110_1);  add_104 = arg110_1 = None
    add_105: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_56, arg111_1);  mul_56 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_280: "f32[128, 128]" = torch.ops.aten.reshape.default(add_105, [128, 128]);  add_105 = None
    permute_140: "f32[128, 512]" = torch.ops.aten.permute.default(arg602_1, [1, 0]);  arg602_1 = None
    
    # No stacktrace found for following nodes
    mm_default_256: "f32[128, 512]" = torch.ops.aten.mm.default(view_280, permute_140);  view_280 = permute_140 = None
    add_tensor_256: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_256, arg603_1);  mm_default_256 = arg603_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_281: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_256, [1, 128, 512]);  add_tensor_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_106: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_281, add_92);  view_281 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_106, arg112_1);  add_106 = arg112_1 = None
    add_107: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_57, arg113_1);  mul_57 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_284: "f32[128, 512]" = torch.ops.aten.reshape.default(add_107, [128, 512])
    permute_142: "f32[512, 128]" = torch.ops.aten.permute.default(arg606_1, [1, 0]);  arg606_1 = None
    
    # No stacktrace found for following nodes
    mm_default_255: "f32[128, 128]" = torch.ops.aten.mm.default(view_284, permute_142);  view_284 = permute_142 = None
    add_tensor_255: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_255, arg607_1);  mm_default_255 = arg607_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_285: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_255, [1, 128, 128]);  add_tensor_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_59: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_285, arg116_1);  view_285 = arg116_1 = None
    add_109: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, arg117_1);  mul_59 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_286: "f32[128, 128]" = torch.ops.aten.reshape.default(add_109, [128, 128])
    permute_143: "f32[128, 128]" = torch.ops.aten.permute.default(arg608_1, [1, 0]);  arg608_1 = None
    
    # No stacktrace found for following nodes
    mm_default_254: "f32[128, 128]" = torch.ops.aten.mm.default(view_286, permute_143);  view_286 = permute_143 = None
    add_tensor_254: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_254, arg609_1);  mm_default_254 = arg609_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_287: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_254, [1, 128, 128]);  add_tensor_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_292: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_287, [1, 128, 4, 32]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_146: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # No stacktrace found for following nodes
    clone_default_48: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_288: "f32[128, 128]" = torch.ops.aten.reshape.default(add_109, [128, 128]);  add_109 = None
    permute_144: "f32[128, 128]" = torch.ops.aten.permute.default(arg610_1, [1, 0]);  arg610_1 = None
    
    # No stacktrace found for following nodes
    mm_default_253: "f32[128, 128]" = torch.ops.aten.mm.default(view_288, permute_144);  view_288 = permute_144 = None
    add_tensor_253: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_253, arg611_1);  mm_default_253 = arg611_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_289: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_253, [1, 128, 128]);  add_tensor_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_293: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_289, [1, 128, 4, 32]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_147: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    
    # No stacktrace found for following nodes
    clone_default_49: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_290: "f32[128, 512]" = torch.ops.aten.reshape.default(add_107, [128, 512])
    permute_145: "f32[512, 128]" = torch.ops.aten.permute.default(arg612_1, [1, 0]);  arg612_1 = None
    
    # No stacktrace found for following nodes
    mm_default_252: "f32[128, 128]" = torch.ops.aten.mm.default(view_290, permute_145);  view_290 = permute_145 = None
    add_tensor_252: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_252, arg613_1);  mm_default_252 = arg613_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_291: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_252, [1, 128, 128]);  add_tensor_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_294: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_291, [1, 128, 4, 32]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_148: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # No stacktrace found for following nodes
    clone_default_50: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_48, clone_default_49, clone_default_50, None, False, scale = 0.17677669529663687);  clone_default_48 = clone_default_49 = clone_default_50 = None
    getitem_18: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_16[0];  _scaled_dot_product_efficient_attention_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_301: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_150, [1, 128, 128]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_302: "f32[128, 128]" = torch.ops.aten.reshape.default(view_301, [128, 128]);  view_301 = None
    permute_151: "f32[128, 128]" = torch.ops.aten.permute.default(arg614_1, [1, 0]);  arg614_1 = None
    
    # No stacktrace found for following nodes
    mm_default_251: "f32[128, 128]" = torch.ops.aten.mm.default(view_302, permute_151);  view_302 = permute_151 = None
    add_tensor_251: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_251, arg615_1);  mm_default_251 = arg615_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_303: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_251, [1, 128, 128]);  add_tensor_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_282: "f32[128, 512]" = torch.ops.aten.reshape.default(add_107, [128, 512])
    permute_141: "f32[512, 128]" = torch.ops.aten.permute.default(arg604_1, [1, 0]);  arg604_1 = None
    
    # No stacktrace found for following nodes
    mm_default_250: "f32[128, 128]" = torch.ops.aten.mm.default(view_282, permute_141);  view_282 = permute_141 = None
    add_tensor_250: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_250, arg605_1);  mm_default_250 = arg605_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_283: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_250, [1, 128, 128]);  add_tensor_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_58: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_283, arg114_1);  view_283 = arg114_1 = None
    add_108: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_58, arg115_1);  mul_58 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_111: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_303, add_108);  view_303 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_60: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_111, arg118_1);  add_111 = arg118_1 = None
    add_112: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_60, arg119_1);  mul_60 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_304: "f32[128, 128]" = torch.ops.aten.reshape.default(add_112, [128, 128])
    permute_152: "f32[128, 512]" = torch.ops.aten.permute.default(arg616_1, [1, 0]);  arg616_1 = None
    
    # No stacktrace found for following nodes
    mm_default_249: "f32[128, 512]" = torch.ops.aten.mm.default(view_304, permute_152);  view_304 = permute_152 = None
    add_tensor_249: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_249, arg617_1);  mm_default_249 = arg617_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_249, [1, 128, 512]);  add_tensor_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_28: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_305);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_306: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_28, [128, 512]);  relu_28 = None
    permute_153: "f32[512, 128]" = torch.ops.aten.permute.default(arg618_1, [1, 0]);  arg618_1 = None
    
    # No stacktrace found for following nodes
    mm_default_248: "f32[128, 128]" = torch.ops.aten.mm.default(view_306, permute_153);  view_306 = permute_153 = None
    add_tensor_248: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_248, arg619_1);  mm_default_248 = arg619_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_307: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_248, [1, 128, 128]);  add_tensor_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_113: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_307, add_112);  view_307 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_61: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_113, arg120_1);  add_113 = arg120_1 = None
    add_114: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, arg121_1);  mul_61 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_308: "f32[128, 128]" = torch.ops.aten.reshape.default(add_114, [128, 128])
    permute_154: "f32[128, 512]" = torch.ops.aten.permute.default(arg620_1, [1, 0]);  arg620_1 = None
    
    # No stacktrace found for following nodes
    mm_default_247: "f32[128, 512]" = torch.ops.aten.mm.default(view_308, permute_154);  view_308 = permute_154 = None
    add_tensor_247: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_247, arg621_1);  mm_default_247 = arg621_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_309: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_247, [1, 128, 512]);  add_tensor_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_29: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_310: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_29, [128, 512]);  relu_29 = None
    permute_155: "f32[512, 128]" = torch.ops.aten.permute.default(arg622_1, [1, 0]);  arg622_1 = None
    
    # No stacktrace found for following nodes
    mm_default_246: "f32[128, 128]" = torch.ops.aten.mm.default(view_310, permute_155);  view_310 = permute_155 = None
    add_tensor_246: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_246, arg623_1);  mm_default_246 = arg623_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_311: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_246, [1, 128, 128]);  add_tensor_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_115: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_311, add_114);  view_311 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_62: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_115, arg122_1);  add_115 = arg122_1 = None
    add_116: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_62, arg123_1);  mul_62 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_312: "f32[128, 128]" = torch.ops.aten.reshape.default(add_116, [128, 128])
    permute_156: "f32[128, 512]" = torch.ops.aten.permute.default(arg624_1, [1, 0]);  arg624_1 = None
    
    # No stacktrace found for following nodes
    mm_default_245: "f32[128, 512]" = torch.ops.aten.mm.default(view_312, permute_156);  view_312 = permute_156 = None
    add_tensor_245: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_245, arg625_1);  mm_default_245 = arg625_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_313: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_245, [1, 128, 512]);  add_tensor_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_30: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_313);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_314: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_30, [128, 512]);  relu_30 = None
    permute_157: "f32[512, 128]" = torch.ops.aten.permute.default(arg626_1, [1, 0]);  arg626_1 = None
    
    # No stacktrace found for following nodes
    mm_default_244: "f32[128, 128]" = torch.ops.aten.mm.default(view_314, permute_157);  view_314 = permute_157 = None
    add_tensor_244: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_244, arg627_1);  mm_default_244 = arg627_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_315: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_244, [1, 128, 128]);  add_tensor_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_117: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_315, add_116);  view_315 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_63: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_117, arg124_1);  add_117 = arg124_1 = None
    add_118: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_63, arg125_1);  mul_63 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_316: "f32[128, 128]" = torch.ops.aten.reshape.default(add_118, [128, 128])
    permute_158: "f32[128, 512]" = torch.ops.aten.permute.default(arg628_1, [1, 0]);  arg628_1 = None
    
    # No stacktrace found for following nodes
    mm_default_243: "f32[128, 512]" = torch.ops.aten.mm.default(view_316, permute_158);  view_316 = permute_158 = None
    add_tensor_243: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_243, arg629_1);  mm_default_243 = arg629_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_317: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_243, [1, 128, 512]);  add_tensor_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_31: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_317);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_318: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_31, [128, 512]);  relu_31 = None
    permute_159: "f32[512, 128]" = torch.ops.aten.permute.default(arg630_1, [1, 0]);  arg630_1 = None
    
    # No stacktrace found for following nodes
    mm_default_242: "f32[128, 128]" = torch.ops.aten.mm.default(view_318, permute_159);  view_318 = permute_159 = None
    add_tensor_242: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_242, arg631_1);  mm_default_242 = arg631_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_319: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_242, [1, 128, 128]);  add_tensor_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_119: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_319, add_118);  view_319 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_64: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, arg126_1);  add_119 = arg126_1 = None
    add_120: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_64, arg127_1);  mul_64 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_320: "f32[128, 128]" = torch.ops.aten.reshape.default(add_120, [128, 128]);  add_120 = None
    permute_160: "f32[128, 512]" = torch.ops.aten.permute.default(arg632_1, [1, 0]);  arg632_1 = None
    
    # No stacktrace found for following nodes
    mm_default_241: "f32[128, 512]" = torch.ops.aten.mm.default(view_320, permute_160);  view_320 = permute_160 = None
    add_tensor_241: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_241, arg633_1);  mm_default_241 = arg633_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_321: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_241, [1, 128, 512]);  add_tensor_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_121: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_321, add_107);  view_321 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_121, arg128_1);  add_121 = arg128_1 = None
    add_122: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_65, arg129_1);  mul_65 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_324: "f32[128, 512]" = torch.ops.aten.reshape.default(add_122, [128, 512])
    permute_162: "f32[512, 128]" = torch.ops.aten.permute.default(arg636_1, [1, 0]);  arg636_1 = None
    
    # No stacktrace found for following nodes
    mm_default_240: "f32[128, 128]" = torch.ops.aten.mm.default(view_324, permute_162);  view_324 = permute_162 = None
    add_tensor_240: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_240, arg637_1);  mm_default_240 = arg637_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_325: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_240, [1, 128, 128]);  add_tensor_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_67: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_325, arg132_1);  view_325 = arg132_1 = None
    add_124: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_67, arg133_1);  mul_67 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_326: "f32[128, 128]" = torch.ops.aten.reshape.default(add_124, [128, 128])
    permute_163: "f32[128, 128]" = torch.ops.aten.permute.default(arg638_1, [1, 0]);  arg638_1 = None
    
    # No stacktrace found for following nodes
    mm_default_239: "f32[128, 128]" = torch.ops.aten.mm.default(view_326, permute_163);  view_326 = permute_163 = None
    add_tensor_239: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_239, arg639_1);  mm_default_239 = arg639_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_327: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_239, [1, 128, 128]);  add_tensor_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_332: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_327, [1, 128, 4, 32]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_166: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    
    # No stacktrace found for following nodes
    clone_default_45: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_328: "f32[128, 128]" = torch.ops.aten.reshape.default(add_124, [128, 128]);  add_124 = None
    permute_164: "f32[128, 128]" = torch.ops.aten.permute.default(arg640_1, [1, 0]);  arg640_1 = None
    
    # No stacktrace found for following nodes
    mm_default_238: "f32[128, 128]" = torch.ops.aten.mm.default(view_328, permute_164);  view_328 = permute_164 = None
    add_tensor_238: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_238, arg641_1);  mm_default_238 = arg641_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_329: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_238, [1, 128, 128]);  add_tensor_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_333: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_329, [1, 128, 4, 32]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
    
    # No stacktrace found for following nodes
    clone_default_46: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_330: "f32[128, 512]" = torch.ops.aten.reshape.default(add_122, [128, 512])
    permute_165: "f32[512, 128]" = torch.ops.aten.permute.default(arg642_1, [1, 0]);  arg642_1 = None
    
    # No stacktrace found for following nodes
    mm_default_237: "f32[128, 128]" = torch.ops.aten.mm.default(view_330, permute_165);  view_330 = permute_165 = None
    add_tensor_237: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_237, arg643_1);  mm_default_237 = arg643_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_331: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_237, [1, 128, 128]);  add_tensor_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_334: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_331, [1, 128, 4, 32]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_168: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # No stacktrace found for following nodes
    clone_default_47: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_45, clone_default_46, clone_default_47, None, False, scale = 0.17677669529663687);  clone_default_45 = clone_default_46 = clone_default_47 = None
    getitem_17: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_170: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_17, [0, 2, 1, 3]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_341: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_170, [1, 128, 128]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_342: "f32[128, 128]" = torch.ops.aten.reshape.default(view_341, [128, 128]);  view_341 = None
    permute_171: "f32[128, 128]" = torch.ops.aten.permute.default(arg644_1, [1, 0]);  arg644_1 = None
    
    # No stacktrace found for following nodes
    mm_default_236: "f32[128, 128]" = torch.ops.aten.mm.default(view_342, permute_171);  view_342 = permute_171 = None
    add_tensor_236: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_236, arg645_1);  mm_default_236 = arg645_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_343: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_236, [1, 128, 128]);  add_tensor_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_322: "f32[128, 512]" = torch.ops.aten.reshape.default(add_122, [128, 512])
    permute_161: "f32[512, 128]" = torch.ops.aten.permute.default(arg634_1, [1, 0]);  arg634_1 = None
    
    # No stacktrace found for following nodes
    mm_default_235: "f32[128, 128]" = torch.ops.aten.mm.default(view_322, permute_161);  view_322 = permute_161 = None
    add_tensor_235: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_235, arg635_1);  mm_default_235 = arg635_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_323: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_235, [1, 128, 128]);  add_tensor_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_66: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_323, arg130_1);  view_323 = arg130_1 = None
    add_123: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_66, arg131_1);  mul_66 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_126: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_343, add_123);  view_343 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_68: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_126, arg134_1);  add_126 = arg134_1 = None
    add_127: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_68, arg135_1);  mul_68 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_344: "f32[128, 128]" = torch.ops.aten.reshape.default(add_127, [128, 128])
    permute_172: "f32[128, 512]" = torch.ops.aten.permute.default(arg646_1, [1, 0]);  arg646_1 = None
    
    # No stacktrace found for following nodes
    mm_default_234: "f32[128, 512]" = torch.ops.aten.mm.default(view_344, permute_172);  view_344 = permute_172 = None
    add_tensor_234: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_234, arg647_1);  mm_default_234 = arg647_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_345: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_234, [1, 128, 512]);  add_tensor_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_32: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_345);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_346: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_32, [128, 512]);  relu_32 = None
    permute_173: "f32[512, 128]" = torch.ops.aten.permute.default(arg648_1, [1, 0]);  arg648_1 = None
    
    # No stacktrace found for following nodes
    mm_default_233: "f32[128, 128]" = torch.ops.aten.mm.default(view_346, permute_173);  view_346 = permute_173 = None
    add_tensor_233: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_233, arg649_1);  mm_default_233 = arg649_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_347: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_233, [1, 128, 128]);  add_tensor_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_128: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_347, add_127);  view_347 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_69: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_128, arg136_1);  add_128 = arg136_1 = None
    add_129: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_69, arg137_1);  mul_69 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_348: "f32[128, 128]" = torch.ops.aten.reshape.default(add_129, [128, 128])
    permute_174: "f32[128, 512]" = torch.ops.aten.permute.default(arg650_1, [1, 0]);  arg650_1 = None
    
    # No stacktrace found for following nodes
    mm_default_232: "f32[128, 512]" = torch.ops.aten.mm.default(view_348, permute_174);  view_348 = permute_174 = None
    add_tensor_232: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_232, arg651_1);  mm_default_232 = arg651_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_349: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_232, [1, 128, 512]);  add_tensor_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_33: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_349);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_350: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_33, [128, 512]);  relu_33 = None
    permute_175: "f32[512, 128]" = torch.ops.aten.permute.default(arg652_1, [1, 0]);  arg652_1 = None
    
    # No stacktrace found for following nodes
    mm_default_231: "f32[128, 128]" = torch.ops.aten.mm.default(view_350, permute_175);  view_350 = permute_175 = None
    add_tensor_231: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_231, arg653_1);  mm_default_231 = arg653_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_351: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_231, [1, 128, 128]);  add_tensor_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_130: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_351, add_129);  view_351 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_70: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_130, arg138_1);  add_130 = arg138_1 = None
    add_131: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_70, arg139_1);  mul_70 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_352: "f32[128, 128]" = torch.ops.aten.reshape.default(add_131, [128, 128])
    permute_176: "f32[128, 512]" = torch.ops.aten.permute.default(arg654_1, [1, 0]);  arg654_1 = None
    
    # No stacktrace found for following nodes
    mm_default_230: "f32[128, 512]" = torch.ops.aten.mm.default(view_352, permute_176);  view_352 = permute_176 = None
    add_tensor_230: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_230, arg655_1);  mm_default_230 = arg655_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_353: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_230, [1, 128, 512]);  add_tensor_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_34: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_353);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_354: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_34, [128, 512]);  relu_34 = None
    permute_177: "f32[512, 128]" = torch.ops.aten.permute.default(arg656_1, [1, 0]);  arg656_1 = None
    
    # No stacktrace found for following nodes
    mm_default_229: "f32[128, 128]" = torch.ops.aten.mm.default(view_354, permute_177);  view_354 = permute_177 = None
    add_tensor_229: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_229, arg657_1);  mm_default_229 = arg657_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_355: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_229, [1, 128, 128]);  add_tensor_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_132: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_355, add_131);  view_355 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_71: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_132, arg140_1);  add_132 = arg140_1 = None
    add_133: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_71, arg141_1);  mul_71 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[128, 128]" = torch.ops.aten.reshape.default(add_133, [128, 128])
    permute_178: "f32[128, 512]" = torch.ops.aten.permute.default(arg658_1, [1, 0]);  arg658_1 = None
    
    # No stacktrace found for following nodes
    mm_default_228: "f32[128, 512]" = torch.ops.aten.mm.default(view_356, permute_178);  view_356 = permute_178 = None
    add_tensor_228: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_228, arg659_1);  mm_default_228 = arg659_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_357: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_228, [1, 128, 512]);  add_tensor_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_35: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_357);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_358: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_35, [128, 512]);  relu_35 = None
    permute_179: "f32[512, 128]" = torch.ops.aten.permute.default(arg660_1, [1, 0]);  arg660_1 = None
    
    # No stacktrace found for following nodes
    mm_default_227: "f32[128, 128]" = torch.ops.aten.mm.default(view_358, permute_179);  view_358 = permute_179 = None
    add_tensor_227: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_227, arg661_1);  mm_default_227 = arg661_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_359: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_227, [1, 128, 128]);  add_tensor_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_134: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_133);  view_359 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_72: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_134, arg142_1);  add_134 = arg142_1 = None
    add_135: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_72, arg143_1);  mul_72 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_360: "f32[128, 128]" = torch.ops.aten.reshape.default(add_135, [128, 128]);  add_135 = None
    permute_180: "f32[128, 512]" = torch.ops.aten.permute.default(arg662_1, [1, 0]);  arg662_1 = None
    
    # No stacktrace found for following nodes
    mm_default_226: "f32[128, 512]" = torch.ops.aten.mm.default(view_360, permute_180);  view_360 = permute_180 = None
    add_tensor_226: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_226, arg663_1);  mm_default_226 = arg663_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_361: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_226, [1, 128, 512]);  add_tensor_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_136: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_361, add_122);  view_361 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_73: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_136, arg144_1);  add_136 = arg144_1 = None
    add_137: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_73, arg145_1);  mul_73 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_364: "f32[128, 512]" = torch.ops.aten.reshape.default(add_137, [128, 512])
    permute_182: "f32[512, 128]" = torch.ops.aten.permute.default(arg666_1, [1, 0]);  arg666_1 = None
    
    # No stacktrace found for following nodes
    mm_default_225: "f32[128, 128]" = torch.ops.aten.mm.default(view_364, permute_182);  view_364 = permute_182 = None
    add_tensor_225: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_225, arg667_1);  mm_default_225 = arg667_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_365: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_225, [1, 128, 128]);  add_tensor_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_75: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_365, arg148_1);  view_365 = arg148_1 = None
    add_139: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_75, arg149_1);  mul_75 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_366: "f32[128, 128]" = torch.ops.aten.reshape.default(add_139, [128, 128])
    permute_183: "f32[128, 128]" = torch.ops.aten.permute.default(arg668_1, [1, 0]);  arg668_1 = None
    
    # No stacktrace found for following nodes
    mm_default_224: "f32[128, 128]" = torch.ops.aten.mm.default(view_366, permute_183);  view_366 = permute_183 = None
    add_tensor_224: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_224, arg669_1);  mm_default_224 = arg669_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_367: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_224, [1, 128, 128]);  add_tensor_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_372: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_367, [1, 128, 4, 32]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_186: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # No stacktrace found for following nodes
    clone_default_42: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_368: "f32[128, 128]" = torch.ops.aten.reshape.default(add_139, [128, 128]);  add_139 = None
    permute_184: "f32[128, 128]" = torch.ops.aten.permute.default(arg670_1, [1, 0]);  arg670_1 = None
    
    # No stacktrace found for following nodes
    mm_default_223: "f32[128, 128]" = torch.ops.aten.mm.default(view_368, permute_184);  view_368 = permute_184 = None
    add_tensor_223: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_223, arg671_1);  mm_default_223 = arg671_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_369: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_223, [1, 128, 128]);  add_tensor_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_373: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_369, [1, 128, 4, 32]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_187: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # No stacktrace found for following nodes
    clone_default_43: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_370: "f32[128, 512]" = torch.ops.aten.reshape.default(add_137, [128, 512])
    permute_185: "f32[512, 128]" = torch.ops.aten.permute.default(arg672_1, [1, 0]);  arg672_1 = None
    
    # No stacktrace found for following nodes
    mm_default_222: "f32[128, 128]" = torch.ops.aten.mm.default(view_370, permute_185);  view_370 = permute_185 = None
    add_tensor_222: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_222, arg673_1);  mm_default_222 = arg673_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_371: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_222, [1, 128, 128]);  add_tensor_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_374: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_371, [1, 128, 4, 32]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_188: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # No stacktrace found for following nodes
    clone_default_44: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_42, clone_default_43, clone_default_44, None, False, scale = 0.17677669529663687);  clone_default_42 = clone_default_43 = clone_default_44 = None
    getitem_16: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_190: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_381: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_190, [1, 128, 128]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_382: "f32[128, 128]" = torch.ops.aten.reshape.default(view_381, [128, 128]);  view_381 = None
    permute_191: "f32[128, 128]" = torch.ops.aten.permute.default(arg674_1, [1, 0]);  arg674_1 = None
    
    # No stacktrace found for following nodes
    mm_default_221: "f32[128, 128]" = torch.ops.aten.mm.default(view_382, permute_191);  view_382 = permute_191 = None
    add_tensor_221: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_221, arg675_1);  mm_default_221 = arg675_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_383: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_221, [1, 128, 128]);  add_tensor_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_362: "f32[128, 512]" = torch.ops.aten.reshape.default(add_137, [128, 512])
    permute_181: "f32[512, 128]" = torch.ops.aten.permute.default(arg664_1, [1, 0]);  arg664_1 = None
    
    # No stacktrace found for following nodes
    mm_default_220: "f32[128, 128]" = torch.ops.aten.mm.default(view_362, permute_181);  view_362 = permute_181 = None
    add_tensor_220: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_220, arg665_1);  mm_default_220 = arg665_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_363: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_220, [1, 128, 128]);  add_tensor_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_74: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_363, arg146_1);  view_363 = arg146_1 = None
    add_138: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_74, arg147_1);  mul_74 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_141: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_383, add_138);  view_383 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_76: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_141, arg150_1);  add_141 = arg150_1 = None
    add_142: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_76, arg151_1);  mul_76 = arg151_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_384: "f32[128, 128]" = torch.ops.aten.reshape.default(add_142, [128, 128])
    permute_192: "f32[128, 512]" = torch.ops.aten.permute.default(arg676_1, [1, 0]);  arg676_1 = None
    
    # No stacktrace found for following nodes
    mm_default_219: "f32[128, 512]" = torch.ops.aten.mm.default(view_384, permute_192);  view_384 = permute_192 = None
    add_tensor_219: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_219, arg677_1);  mm_default_219 = arg677_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_385: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_219, [1, 128, 512]);  add_tensor_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_36: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_385);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_386: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_36, [128, 512]);  relu_36 = None
    permute_193: "f32[512, 128]" = torch.ops.aten.permute.default(arg678_1, [1, 0]);  arg678_1 = None
    
    # No stacktrace found for following nodes
    mm_default_218: "f32[128, 128]" = torch.ops.aten.mm.default(view_386, permute_193);  view_386 = permute_193 = None
    add_tensor_218: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_218, arg679_1);  mm_default_218 = arg679_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_387: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_218, [1, 128, 128]);  add_tensor_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_143: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_387, add_142);  view_387 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_77: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_143, arg152_1);  add_143 = arg152_1 = None
    add_144: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_77, arg153_1);  mul_77 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_388: "f32[128, 128]" = torch.ops.aten.reshape.default(add_144, [128, 128])
    permute_194: "f32[128, 512]" = torch.ops.aten.permute.default(arg680_1, [1, 0]);  arg680_1 = None
    
    # No stacktrace found for following nodes
    mm_default_217: "f32[128, 512]" = torch.ops.aten.mm.default(view_388, permute_194);  view_388 = permute_194 = None
    add_tensor_217: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_217, arg681_1);  mm_default_217 = arg681_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_217, [1, 128, 512]);  add_tensor_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_37: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_389);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_390: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_37, [128, 512]);  relu_37 = None
    permute_195: "f32[512, 128]" = torch.ops.aten.permute.default(arg682_1, [1, 0]);  arg682_1 = None
    
    # No stacktrace found for following nodes
    mm_default_216: "f32[128, 128]" = torch.ops.aten.mm.default(view_390, permute_195);  view_390 = permute_195 = None
    add_tensor_216: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_216, arg683_1);  mm_default_216 = arg683_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_391: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_216, [1, 128, 128]);  add_tensor_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_145: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_391, add_144);  view_391 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_78: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_145, arg154_1);  add_145 = arg154_1 = None
    add_146: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_78, arg155_1);  mul_78 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[128, 128]" = torch.ops.aten.reshape.default(add_146, [128, 128])
    permute_196: "f32[128, 512]" = torch.ops.aten.permute.default(arg684_1, [1, 0]);  arg684_1 = None
    
    # No stacktrace found for following nodes
    mm_default_215: "f32[128, 512]" = torch.ops.aten.mm.default(view_392, permute_196);  view_392 = permute_196 = None
    add_tensor_215: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_215, arg685_1);  mm_default_215 = arg685_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_393: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_215, [1, 128, 512]);  add_tensor_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_38: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_393);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_394: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_38, [128, 512]);  relu_38 = None
    permute_197: "f32[512, 128]" = torch.ops.aten.permute.default(arg686_1, [1, 0]);  arg686_1 = None
    
    # No stacktrace found for following nodes
    mm_default_214: "f32[128, 128]" = torch.ops.aten.mm.default(view_394, permute_197);  view_394 = permute_197 = None
    add_tensor_214: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_214, arg687_1);  mm_default_214 = arg687_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_395: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_214, [1, 128, 128]);  add_tensor_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_147: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_395, add_146);  view_395 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_79: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_147, arg156_1);  add_147 = arg156_1 = None
    add_148: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, arg157_1);  mul_79 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_396: "f32[128, 128]" = torch.ops.aten.reshape.default(add_148, [128, 128])
    permute_198: "f32[128, 512]" = torch.ops.aten.permute.default(arg688_1, [1, 0]);  arg688_1 = None
    
    # No stacktrace found for following nodes
    mm_default_213: "f32[128, 512]" = torch.ops.aten.mm.default(view_396, permute_198);  view_396 = permute_198 = None
    add_tensor_213: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_213, arg689_1);  mm_default_213 = arg689_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_397: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_213, [1, 128, 512]);  add_tensor_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_39: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_397);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_398: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_39, [128, 512]);  relu_39 = None
    permute_199: "f32[512, 128]" = torch.ops.aten.permute.default(arg690_1, [1, 0]);  arg690_1 = None
    
    # No stacktrace found for following nodes
    mm_default_212: "f32[128, 128]" = torch.ops.aten.mm.default(view_398, permute_199);  view_398 = permute_199 = None
    add_tensor_212: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_212, arg691_1);  mm_default_212 = arg691_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_399: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_212, [1, 128, 128]);  add_tensor_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_149: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_399, add_148);  view_399 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_80: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_149, arg158_1);  add_149 = arg158_1 = None
    add_150: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_80, arg159_1);  mul_80 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_400: "f32[128, 128]" = torch.ops.aten.reshape.default(add_150, [128, 128]);  add_150 = None
    permute_200: "f32[128, 512]" = torch.ops.aten.permute.default(arg692_1, [1, 0]);  arg692_1 = None
    
    # No stacktrace found for following nodes
    mm_default_211: "f32[128, 512]" = torch.ops.aten.mm.default(view_400, permute_200);  view_400 = permute_200 = None
    add_tensor_211: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_211, arg693_1);  mm_default_211 = arg693_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_401: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_211, [1, 128, 512]);  add_tensor_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_151: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_401, add_137);  view_401 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_81: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_151, arg160_1);  add_151 = arg160_1 = None
    add_152: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_81, arg161_1);  mul_81 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_404: "f32[128, 512]" = torch.ops.aten.reshape.default(add_152, [128, 512])
    permute_202: "f32[512, 128]" = torch.ops.aten.permute.default(arg696_1, [1, 0]);  arg696_1 = None
    
    # No stacktrace found for following nodes
    mm_default_210: "f32[128, 128]" = torch.ops.aten.mm.default(view_404, permute_202);  view_404 = permute_202 = None
    add_tensor_210: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_210, arg697_1);  mm_default_210 = arg697_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_405: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_210, [1, 128, 128]);  add_tensor_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_83: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_405, arg164_1);  view_405 = arg164_1 = None
    add_154: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_83, arg165_1);  mul_83 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_406: "f32[128, 128]" = torch.ops.aten.reshape.default(add_154, [128, 128])
    permute_203: "f32[128, 128]" = torch.ops.aten.permute.default(arg698_1, [1, 0]);  arg698_1 = None
    
    # No stacktrace found for following nodes
    mm_default_209: "f32[128, 128]" = torch.ops.aten.mm.default(view_406, permute_203);  view_406 = permute_203 = None
    add_tensor_209: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_209, arg699_1);  mm_default_209 = arg699_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_407: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_209, [1, 128, 128]);  add_tensor_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_412: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_407, [1, 128, 4, 32]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_206: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    
    # No stacktrace found for following nodes
    clone_default_39: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_408: "f32[128, 128]" = torch.ops.aten.reshape.default(add_154, [128, 128]);  add_154 = None
    permute_204: "f32[128, 128]" = torch.ops.aten.permute.default(arg700_1, [1, 0]);  arg700_1 = None
    
    # No stacktrace found for following nodes
    mm_default_208: "f32[128, 128]" = torch.ops.aten.mm.default(view_408, permute_204);  view_408 = permute_204 = None
    add_tensor_208: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_208, arg701_1);  mm_default_208 = arg701_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_409: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_208, [1, 128, 128]);  add_tensor_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_413: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_409, [1, 128, 4, 32]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_207: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    
    # No stacktrace found for following nodes
    clone_default_40: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_410: "f32[128, 512]" = torch.ops.aten.reshape.default(add_152, [128, 512])
    permute_205: "f32[512, 128]" = torch.ops.aten.permute.default(arg702_1, [1, 0]);  arg702_1 = None
    
    # No stacktrace found for following nodes
    mm_default_207: "f32[128, 128]" = torch.ops.aten.mm.default(view_410, permute_205);  view_410 = permute_205 = None
    add_tensor_207: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_207, arg703_1);  mm_default_207 = arg703_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_411: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_207, [1, 128, 128]);  add_tensor_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_414: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_411, [1, 128, 4, 32]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_208: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
    
    # No stacktrace found for following nodes
    clone_default_41: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_39, clone_default_40, clone_default_41, None, False, scale = 0.17677669529663687);  clone_default_39 = clone_default_40 = clone_default_41 = None
    getitem_15: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_210: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_421: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_210, [1, 128, 128]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_422: "f32[128, 128]" = torch.ops.aten.reshape.default(view_421, [128, 128]);  view_421 = None
    permute_211: "f32[128, 128]" = torch.ops.aten.permute.default(arg704_1, [1, 0]);  arg704_1 = None
    
    # No stacktrace found for following nodes
    mm_default_206: "f32[128, 128]" = torch.ops.aten.mm.default(view_422, permute_211);  view_422 = permute_211 = None
    add_tensor_206: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_206, arg705_1);  mm_default_206 = arg705_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_423: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_206, [1, 128, 128]);  add_tensor_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_402: "f32[128, 512]" = torch.ops.aten.reshape.default(add_152, [128, 512])
    permute_201: "f32[512, 128]" = torch.ops.aten.permute.default(arg694_1, [1, 0]);  arg694_1 = None
    
    # No stacktrace found for following nodes
    mm_default_205: "f32[128, 128]" = torch.ops.aten.mm.default(view_402, permute_201);  view_402 = permute_201 = None
    add_tensor_205: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_205, arg695_1);  mm_default_205 = arg695_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_403: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_205, [1, 128, 128]);  add_tensor_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_82: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_403, arg162_1);  view_403 = arg162_1 = None
    add_153: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_82, arg163_1);  mul_82 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_156: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_423, add_153);  view_423 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_84: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_156, arg166_1);  add_156 = arg166_1 = None
    add_157: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_84, arg167_1);  mul_84 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_424: "f32[128, 128]" = torch.ops.aten.reshape.default(add_157, [128, 128])
    permute_212: "f32[128, 512]" = torch.ops.aten.permute.default(arg706_1, [1, 0]);  arg706_1 = None
    
    # No stacktrace found for following nodes
    mm_default_204: "f32[128, 512]" = torch.ops.aten.mm.default(view_424, permute_212);  view_424 = permute_212 = None
    add_tensor_204: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_204, arg707_1);  mm_default_204 = arg707_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_425: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_204, [1, 128, 512]);  add_tensor_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_40: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_425);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_426: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_40, [128, 512]);  relu_40 = None
    permute_213: "f32[512, 128]" = torch.ops.aten.permute.default(arg708_1, [1, 0]);  arg708_1 = None
    
    # No stacktrace found for following nodes
    mm_default_203: "f32[128, 128]" = torch.ops.aten.mm.default(view_426, permute_213);  view_426 = permute_213 = None
    add_tensor_203: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_203, arg709_1);  mm_default_203 = arg709_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_427: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_203, [1, 128, 128]);  add_tensor_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_158: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_427, add_157);  view_427 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_85: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_158, arg168_1);  add_158 = arg168_1 = None
    add_159: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_85, arg169_1);  mul_85 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_428: "f32[128, 128]" = torch.ops.aten.reshape.default(add_159, [128, 128])
    permute_214: "f32[128, 512]" = torch.ops.aten.permute.default(arg710_1, [1, 0]);  arg710_1 = None
    
    # No stacktrace found for following nodes
    mm_default_202: "f32[128, 512]" = torch.ops.aten.mm.default(view_428, permute_214);  view_428 = permute_214 = None
    add_tensor_202: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_202, arg711_1);  mm_default_202 = arg711_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_429: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_202, [1, 128, 512]);  add_tensor_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_41: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_429);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_430: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_41, [128, 512]);  relu_41 = None
    permute_215: "f32[512, 128]" = torch.ops.aten.permute.default(arg712_1, [1, 0]);  arg712_1 = None
    
    # No stacktrace found for following nodes
    mm_default_201: "f32[128, 128]" = torch.ops.aten.mm.default(view_430, permute_215);  view_430 = permute_215 = None
    add_tensor_201: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_201, arg713_1);  mm_default_201 = arg713_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_431: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_201, [1, 128, 128]);  add_tensor_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_160: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_431, add_159);  view_431 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_86: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_160, arg170_1);  add_160 = arg170_1 = None
    add_161: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, arg171_1);  mul_86 = arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_432: "f32[128, 128]" = torch.ops.aten.reshape.default(add_161, [128, 128])
    permute_216: "f32[128, 512]" = torch.ops.aten.permute.default(arg714_1, [1, 0]);  arg714_1 = None
    
    # No stacktrace found for following nodes
    mm_default_200: "f32[128, 512]" = torch.ops.aten.mm.default(view_432, permute_216);  view_432 = permute_216 = None
    add_tensor_200: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_200, arg715_1);  mm_default_200 = arg715_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_433: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_200, [1, 128, 512]);  add_tensor_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_42: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_433);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_434: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_42, [128, 512]);  relu_42 = None
    permute_217: "f32[512, 128]" = torch.ops.aten.permute.default(arg716_1, [1, 0]);  arg716_1 = None
    
    # No stacktrace found for following nodes
    mm_default_199: "f32[128, 128]" = torch.ops.aten.mm.default(view_434, permute_217);  view_434 = permute_217 = None
    add_tensor_199: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_199, arg717_1);  mm_default_199 = arg717_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_435: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_199, [1, 128, 128]);  add_tensor_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_162: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_435, add_161);  view_435 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_87: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_162, arg172_1);  add_162 = arg172_1 = None
    add_163: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_87, arg173_1);  mul_87 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_436: "f32[128, 128]" = torch.ops.aten.reshape.default(add_163, [128, 128])
    permute_218: "f32[128, 512]" = torch.ops.aten.permute.default(arg718_1, [1, 0]);  arg718_1 = None
    
    # No stacktrace found for following nodes
    mm_default_198: "f32[128, 512]" = torch.ops.aten.mm.default(view_436, permute_218);  view_436 = permute_218 = None
    add_tensor_198: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_198, arg719_1);  mm_default_198 = arg719_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_437: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_198, [1, 128, 512]);  add_tensor_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_43: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_437);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_438: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_43, [128, 512]);  relu_43 = None
    permute_219: "f32[512, 128]" = torch.ops.aten.permute.default(arg720_1, [1, 0]);  arg720_1 = None
    
    # No stacktrace found for following nodes
    mm_default_197: "f32[128, 128]" = torch.ops.aten.mm.default(view_438, permute_219);  view_438 = permute_219 = None
    add_tensor_197: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_197, arg721_1);  mm_default_197 = arg721_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_439: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_197, [1, 128, 128]);  add_tensor_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_164: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_439, add_163);  view_439 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_88: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_164, arg174_1);  add_164 = arg174_1 = None
    add_165: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_88, arg175_1);  mul_88 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_440: "f32[128, 128]" = torch.ops.aten.reshape.default(add_165, [128, 128]);  add_165 = None
    permute_220: "f32[128, 512]" = torch.ops.aten.permute.default(arg722_1, [1, 0]);  arg722_1 = None
    
    # No stacktrace found for following nodes
    mm_default_196: "f32[128, 512]" = torch.ops.aten.mm.default(view_440, permute_220);  view_440 = permute_220 = None
    add_tensor_196: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_196, arg723_1);  mm_default_196 = arg723_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_441: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_196, [1, 128, 512]);  add_tensor_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_166: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_441, add_152);  view_441 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_89: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_166, arg176_1);  add_166 = arg176_1 = None
    add_167: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_89, arg177_1);  mul_89 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_444: "f32[128, 512]" = torch.ops.aten.reshape.default(add_167, [128, 512])
    permute_222: "f32[512, 128]" = torch.ops.aten.permute.default(arg726_1, [1, 0]);  arg726_1 = None
    
    # No stacktrace found for following nodes
    mm_default_195: "f32[128, 128]" = torch.ops.aten.mm.default(view_444, permute_222);  view_444 = permute_222 = None
    add_tensor_195: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_195, arg727_1);  mm_default_195 = arg727_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_445: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_195, [1, 128, 128]);  add_tensor_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_91: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_445, arg180_1);  view_445 = arg180_1 = None
    add_169: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_91, arg181_1);  mul_91 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_446: "f32[128, 128]" = torch.ops.aten.reshape.default(add_169, [128, 128])
    permute_223: "f32[128, 128]" = torch.ops.aten.permute.default(arg728_1, [1, 0]);  arg728_1 = None
    
    # No stacktrace found for following nodes
    mm_default_194: "f32[128, 128]" = torch.ops.aten.mm.default(view_446, permute_223);  view_446 = permute_223 = None
    add_tensor_194: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_194, arg729_1);  mm_default_194 = arg729_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_447: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_194, [1, 128, 128]);  add_tensor_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_452: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_447, [1, 128, 4, 32]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_226: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
    
    # No stacktrace found for following nodes
    clone_default_36: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_448: "f32[128, 128]" = torch.ops.aten.reshape.default(add_169, [128, 128]);  add_169 = None
    permute_224: "f32[128, 128]" = torch.ops.aten.permute.default(arg730_1, [1, 0]);  arg730_1 = None
    
    # No stacktrace found for following nodes
    mm_default_193: "f32[128, 128]" = torch.ops.aten.mm.default(view_448, permute_224);  view_448 = permute_224 = None
    add_tensor_193: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_193, arg731_1);  mm_default_193 = arg731_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_449: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_193, [1, 128, 128]);  add_tensor_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_453: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_449, [1, 128, 4, 32]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # No stacktrace found for following nodes
    clone_default_37: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_450: "f32[128, 512]" = torch.ops.aten.reshape.default(add_167, [128, 512])
    permute_225: "f32[512, 128]" = torch.ops.aten.permute.default(arg732_1, [1, 0]);  arg732_1 = None
    
    # No stacktrace found for following nodes
    mm_default_192: "f32[128, 128]" = torch.ops.aten.mm.default(view_450, permute_225);  view_450 = permute_225 = None
    add_tensor_192: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_192, arg733_1);  mm_default_192 = arg733_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_451: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_192, [1, 128, 128]);  add_tensor_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_454: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_451, [1, 128, 4, 32]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # No stacktrace found for following nodes
    clone_default_38: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_36, clone_default_37, clone_default_38, None, False, scale = 0.17677669529663687);  clone_default_36 = clone_default_37 = clone_default_38 = None
    getitem_14: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_230: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3]);  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_461: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_230, [1, 128, 128]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_462: "f32[128, 128]" = torch.ops.aten.reshape.default(view_461, [128, 128]);  view_461 = None
    permute_231: "f32[128, 128]" = torch.ops.aten.permute.default(arg734_1, [1, 0]);  arg734_1 = None
    
    # No stacktrace found for following nodes
    mm_default_191: "f32[128, 128]" = torch.ops.aten.mm.default(view_462, permute_231);  view_462 = permute_231 = None
    add_tensor_191: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_191, arg735_1);  mm_default_191 = arg735_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_463: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_191, [1, 128, 128]);  add_tensor_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_442: "f32[128, 512]" = torch.ops.aten.reshape.default(add_167, [128, 512])
    permute_221: "f32[512, 128]" = torch.ops.aten.permute.default(arg724_1, [1, 0]);  arg724_1 = None
    
    # No stacktrace found for following nodes
    mm_default_190: "f32[128, 128]" = torch.ops.aten.mm.default(view_442, permute_221);  view_442 = permute_221 = None
    add_tensor_190: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_190, arg725_1);  mm_default_190 = arg725_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_443: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_190, [1, 128, 128]);  add_tensor_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_90: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_443, arg178_1);  view_443 = arg178_1 = None
    add_168: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_90, arg179_1);  mul_90 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_171: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_463, add_168);  view_463 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_92: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_171, arg182_1);  add_171 = arg182_1 = None
    add_172: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_92, arg183_1);  mul_92 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_464: "f32[128, 128]" = torch.ops.aten.reshape.default(add_172, [128, 128])
    permute_232: "f32[128, 512]" = torch.ops.aten.permute.default(arg736_1, [1, 0]);  arg736_1 = None
    
    # No stacktrace found for following nodes
    mm_default_189: "f32[128, 512]" = torch.ops.aten.mm.default(view_464, permute_232);  view_464 = permute_232 = None
    add_tensor_189: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_189, arg737_1);  mm_default_189 = arg737_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_465: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_189, [1, 128, 512]);  add_tensor_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_44: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_465);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_466: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_44, [128, 512]);  relu_44 = None
    permute_233: "f32[512, 128]" = torch.ops.aten.permute.default(arg738_1, [1, 0]);  arg738_1 = None
    
    # No stacktrace found for following nodes
    mm_default_188: "f32[128, 128]" = torch.ops.aten.mm.default(view_466, permute_233);  view_466 = permute_233 = None
    add_tensor_188: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_188, arg739_1);  mm_default_188 = arg739_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_467: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_188, [1, 128, 128]);  add_tensor_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_173: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_467, add_172);  view_467 = add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_93: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_173, arg184_1);  add_173 = arg184_1 = None
    add_174: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_93, arg185_1);  mul_93 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_468: "f32[128, 128]" = torch.ops.aten.reshape.default(add_174, [128, 128])
    permute_234: "f32[128, 512]" = torch.ops.aten.permute.default(arg740_1, [1, 0]);  arg740_1 = None
    
    # No stacktrace found for following nodes
    mm_default_187: "f32[128, 512]" = torch.ops.aten.mm.default(view_468, permute_234);  view_468 = permute_234 = None
    add_tensor_187: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_187, arg741_1);  mm_default_187 = arg741_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_469: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_187, [1, 128, 512]);  add_tensor_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_45: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_469);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_470: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_45, [128, 512]);  relu_45 = None
    permute_235: "f32[512, 128]" = torch.ops.aten.permute.default(arg742_1, [1, 0]);  arg742_1 = None
    
    # No stacktrace found for following nodes
    mm_default_186: "f32[128, 128]" = torch.ops.aten.mm.default(view_470, permute_235);  view_470 = permute_235 = None
    add_tensor_186: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_186, arg743_1);  mm_default_186 = arg743_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_471: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_186, [1, 128, 128]);  add_tensor_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_175: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_471, add_174);  view_471 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_94: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_175, arg186_1);  add_175 = arg186_1 = None
    add_176: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_94, arg187_1);  mul_94 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_472: "f32[128, 128]" = torch.ops.aten.reshape.default(add_176, [128, 128])
    permute_236: "f32[128, 512]" = torch.ops.aten.permute.default(arg744_1, [1, 0]);  arg744_1 = None
    
    # No stacktrace found for following nodes
    mm_default_185: "f32[128, 512]" = torch.ops.aten.mm.default(view_472, permute_236);  view_472 = permute_236 = None
    add_tensor_185: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_185, arg745_1);  mm_default_185 = arg745_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_185, [1, 128, 512]);  add_tensor_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_46: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_473);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_474: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_46, [128, 512]);  relu_46 = None
    permute_237: "f32[512, 128]" = torch.ops.aten.permute.default(arg746_1, [1, 0]);  arg746_1 = None
    
    # No stacktrace found for following nodes
    mm_default_184: "f32[128, 128]" = torch.ops.aten.mm.default(view_474, permute_237);  view_474 = permute_237 = None
    add_tensor_184: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_184, arg747_1);  mm_default_184 = arg747_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_475: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_184, [1, 128, 128]);  add_tensor_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_177: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_475, add_176);  view_475 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_95: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_177, arg188_1);  add_177 = arg188_1 = None
    add_178: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_95, arg189_1);  mul_95 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[128, 128]" = torch.ops.aten.reshape.default(add_178, [128, 128])
    permute_238: "f32[128, 512]" = torch.ops.aten.permute.default(arg748_1, [1, 0]);  arg748_1 = None
    
    # No stacktrace found for following nodes
    mm_default_183: "f32[128, 512]" = torch.ops.aten.mm.default(view_476, permute_238);  view_476 = permute_238 = None
    add_tensor_183: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_183, arg749_1);  mm_default_183 = arg749_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_477: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_183, [1, 128, 512]);  add_tensor_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_47: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_477);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_478: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_47, [128, 512]);  relu_47 = None
    permute_239: "f32[512, 128]" = torch.ops.aten.permute.default(arg750_1, [1, 0]);  arg750_1 = None
    
    # No stacktrace found for following nodes
    mm_default_182: "f32[128, 128]" = torch.ops.aten.mm.default(view_478, permute_239);  view_478 = permute_239 = None
    add_tensor_182: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_182, arg751_1);  mm_default_182 = arg751_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_479: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_182, [1, 128, 128]);  add_tensor_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_179: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_479, add_178);  view_479 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_96: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_179, arg190_1);  add_179 = arg190_1 = None
    add_180: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_96, arg191_1);  mul_96 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_480: "f32[128, 128]" = torch.ops.aten.reshape.default(add_180, [128, 128]);  add_180 = None
    permute_240: "f32[128, 512]" = torch.ops.aten.permute.default(arg752_1, [1, 0]);  arg752_1 = None
    
    # No stacktrace found for following nodes
    mm_default_181: "f32[128, 512]" = torch.ops.aten.mm.default(view_480, permute_240);  view_480 = permute_240 = None
    add_tensor_181: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_181, arg753_1);  mm_default_181 = arg753_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_481: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_181, [1, 128, 512]);  add_tensor_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_181: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_481, add_167);  view_481 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_97: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_181, arg192_1);  add_181 = arg192_1 = None
    add_182: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_97, arg193_1);  mul_97 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_484: "f32[128, 512]" = torch.ops.aten.reshape.default(add_182, [128, 512])
    permute_242: "f32[512, 128]" = torch.ops.aten.permute.default(arg756_1, [1, 0]);  arg756_1 = None
    
    # No stacktrace found for following nodes
    mm_default_180: "f32[128, 128]" = torch.ops.aten.mm.default(view_484, permute_242);  view_484 = permute_242 = None
    add_tensor_180: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_180, arg757_1);  mm_default_180 = arg757_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_485: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_180, [1, 128, 128]);  add_tensor_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_99: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_485, arg196_1);  view_485 = arg196_1 = None
    add_184: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_99, arg197_1);  mul_99 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_486: "f32[128, 128]" = torch.ops.aten.reshape.default(add_184, [128, 128])
    permute_243: "f32[128, 128]" = torch.ops.aten.permute.default(arg758_1, [1, 0]);  arg758_1 = None
    
    # No stacktrace found for following nodes
    mm_default_179: "f32[128, 128]" = torch.ops.aten.mm.default(view_486, permute_243);  view_486 = permute_243 = None
    add_tensor_179: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_179, arg759_1);  mm_default_179 = arg759_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_487: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_179, [1, 128, 128]);  add_tensor_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_492: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_487, [1, 128, 4, 32]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_246: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # No stacktrace found for following nodes
    clone_default_33: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_488: "f32[128, 128]" = torch.ops.aten.reshape.default(add_184, [128, 128]);  add_184 = None
    permute_244: "f32[128, 128]" = torch.ops.aten.permute.default(arg760_1, [1, 0]);  arg760_1 = None
    
    # No stacktrace found for following nodes
    mm_default_178: "f32[128, 128]" = torch.ops.aten.mm.default(view_488, permute_244);  view_488 = permute_244 = None
    add_tensor_178: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_178, arg761_1);  mm_default_178 = arg761_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_489: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_178, [1, 128, 128]);  add_tensor_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_493: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_489, [1, 128, 4, 32]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_247: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # No stacktrace found for following nodes
    clone_default_34: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_490: "f32[128, 512]" = torch.ops.aten.reshape.default(add_182, [128, 512])
    permute_245: "f32[512, 128]" = torch.ops.aten.permute.default(arg762_1, [1, 0]);  arg762_1 = None
    
    # No stacktrace found for following nodes
    mm_default_177: "f32[128, 128]" = torch.ops.aten.mm.default(view_490, permute_245);  view_490 = permute_245 = None
    add_tensor_177: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_177, arg763_1);  mm_default_177 = arg763_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_491: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_177, [1, 128, 128]);  add_tensor_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_494: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_491, [1, 128, 4, 32]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_248: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # No stacktrace found for following nodes
    clone_default_35: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_33, clone_default_34, clone_default_35, None, False, scale = 0.17677669529663687);  clone_default_33 = clone_default_34 = clone_default_35 = None
    getitem_13: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_250: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3]);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_501: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_250, [1, 128, 128]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_502: "f32[128, 128]" = torch.ops.aten.reshape.default(view_501, [128, 128]);  view_501 = None
    permute_251: "f32[128, 128]" = torch.ops.aten.permute.default(arg764_1, [1, 0]);  arg764_1 = None
    
    # No stacktrace found for following nodes
    mm_default_176: "f32[128, 128]" = torch.ops.aten.mm.default(view_502, permute_251);  view_502 = permute_251 = None
    add_tensor_176: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_176, arg765_1);  mm_default_176 = arg765_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_503: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_176, [1, 128, 128]);  add_tensor_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_482: "f32[128, 512]" = torch.ops.aten.reshape.default(add_182, [128, 512])
    permute_241: "f32[512, 128]" = torch.ops.aten.permute.default(arg754_1, [1, 0]);  arg754_1 = None
    
    # No stacktrace found for following nodes
    mm_default_175: "f32[128, 128]" = torch.ops.aten.mm.default(view_482, permute_241);  view_482 = permute_241 = None
    add_tensor_175: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_175, arg755_1);  mm_default_175 = arg755_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_483: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_175, [1, 128, 128]);  add_tensor_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_98: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_483, arg194_1);  view_483 = arg194_1 = None
    add_183: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_98, arg195_1);  mul_98 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_186: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_503, add_183);  view_503 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_100: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_186, arg198_1);  add_186 = arg198_1 = None
    add_187: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_100, arg199_1);  mul_100 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[128, 128]" = torch.ops.aten.reshape.default(add_187, [128, 128])
    permute_252: "f32[128, 512]" = torch.ops.aten.permute.default(arg766_1, [1, 0]);  arg766_1 = None
    
    # No stacktrace found for following nodes
    mm_default_174: "f32[128, 512]" = torch.ops.aten.mm.default(view_504, permute_252);  view_504 = permute_252 = None
    add_tensor_174: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_174, arg767_1);  mm_default_174 = arg767_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_505: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_174, [1, 128, 512]);  add_tensor_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_48: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_505);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_506: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_48, [128, 512]);  relu_48 = None
    permute_253: "f32[512, 128]" = torch.ops.aten.permute.default(arg768_1, [1, 0]);  arg768_1 = None
    
    # No stacktrace found for following nodes
    mm_default_173: "f32[128, 128]" = torch.ops.aten.mm.default(view_506, permute_253);  view_506 = permute_253 = None
    add_tensor_173: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_173, arg769_1);  mm_default_173 = arg769_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_507: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_173, [1, 128, 128]);  add_tensor_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_188: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_507, add_187);  view_507 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_101: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_188, arg200_1);  add_188 = arg200_1 = None
    add_189: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, arg201_1);  mul_101 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_508: "f32[128, 128]" = torch.ops.aten.reshape.default(add_189, [128, 128])
    permute_254: "f32[128, 512]" = torch.ops.aten.permute.default(arg770_1, [1, 0]);  arg770_1 = None
    
    # No stacktrace found for following nodes
    mm_default_172: "f32[128, 512]" = torch.ops.aten.mm.default(view_508, permute_254);  view_508 = permute_254 = None
    add_tensor_172: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_172, arg771_1);  mm_default_172 = arg771_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_509: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_172, [1, 128, 512]);  add_tensor_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_49: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_509);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_510: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_49, [128, 512]);  relu_49 = None
    permute_255: "f32[512, 128]" = torch.ops.aten.permute.default(arg772_1, [1, 0]);  arg772_1 = None
    
    # No stacktrace found for following nodes
    mm_default_171: "f32[128, 128]" = torch.ops.aten.mm.default(view_510, permute_255);  view_510 = permute_255 = None
    add_tensor_171: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_171, arg773_1);  mm_default_171 = arg773_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_511: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_171, [1, 128, 128]);  add_tensor_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_190: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_511, add_189);  view_511 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_102: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_190, arg202_1);  add_190 = arg202_1 = None
    add_191: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_102, arg203_1);  mul_102 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_512: "f32[128, 128]" = torch.ops.aten.reshape.default(add_191, [128, 128])
    permute_256: "f32[128, 512]" = torch.ops.aten.permute.default(arg774_1, [1, 0]);  arg774_1 = None
    
    # No stacktrace found for following nodes
    mm_default_170: "f32[128, 512]" = torch.ops.aten.mm.default(view_512, permute_256);  view_512 = permute_256 = None
    add_tensor_170: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_170, arg775_1);  mm_default_170 = arg775_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_513: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_170, [1, 128, 512]);  add_tensor_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_50: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_513);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_514: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_50, [128, 512]);  relu_50 = None
    permute_257: "f32[512, 128]" = torch.ops.aten.permute.default(arg776_1, [1, 0]);  arg776_1 = None
    
    # No stacktrace found for following nodes
    mm_default_169: "f32[128, 128]" = torch.ops.aten.mm.default(view_514, permute_257);  view_514 = permute_257 = None
    add_tensor_169: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_169, arg777_1);  mm_default_169 = arg777_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_515: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_169, [1, 128, 128]);  add_tensor_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_192: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_515, add_191);  view_515 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_103: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_192, arg204_1);  add_192 = arg204_1 = None
    add_193: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_103, arg205_1);  mul_103 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_516: "f32[128, 128]" = torch.ops.aten.reshape.default(add_193, [128, 128])
    permute_258: "f32[128, 512]" = torch.ops.aten.permute.default(arg778_1, [1, 0]);  arg778_1 = None
    
    # No stacktrace found for following nodes
    mm_default_168: "f32[128, 512]" = torch.ops.aten.mm.default(view_516, permute_258);  view_516 = permute_258 = None
    add_tensor_168: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_168, arg779_1);  mm_default_168 = arg779_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_517: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_168, [1, 128, 512]);  add_tensor_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_51: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_517);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_518: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_51, [128, 512]);  relu_51 = None
    permute_259: "f32[512, 128]" = torch.ops.aten.permute.default(arg780_1, [1, 0]);  arg780_1 = None
    
    # No stacktrace found for following nodes
    mm_default_167: "f32[128, 128]" = torch.ops.aten.mm.default(view_518, permute_259);  view_518 = permute_259 = None
    add_tensor_167: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_167, arg781_1);  mm_default_167 = arg781_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_519: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_167, [1, 128, 128]);  add_tensor_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_194: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_519, add_193);  view_519 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_104: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_194, arg206_1);  add_194 = arg206_1 = None
    add_195: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_104, arg207_1);  mul_104 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_520: "f32[128, 128]" = torch.ops.aten.reshape.default(add_195, [128, 128]);  add_195 = None
    permute_260: "f32[128, 512]" = torch.ops.aten.permute.default(arg782_1, [1, 0]);  arg782_1 = None
    
    # No stacktrace found for following nodes
    mm_default_166: "f32[128, 512]" = torch.ops.aten.mm.default(view_520, permute_260);  view_520 = permute_260 = None
    add_tensor_166: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_166, arg783_1);  mm_default_166 = arg783_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_521: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_166, [1, 128, 512]);  add_tensor_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_196: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_521, add_182);  view_521 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_105: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_196, arg208_1);  add_196 = arg208_1 = None
    add_197: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_105, arg209_1);  mul_105 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_524: "f32[128, 512]" = torch.ops.aten.reshape.default(add_197, [128, 512])
    permute_262: "f32[512, 128]" = torch.ops.aten.permute.default(arg786_1, [1, 0]);  arg786_1 = None
    
    # No stacktrace found for following nodes
    mm_default_165: "f32[128, 128]" = torch.ops.aten.mm.default(view_524, permute_262);  view_524 = permute_262 = None
    add_tensor_165: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_165, arg787_1);  mm_default_165 = arg787_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_525: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_165, [1, 128, 128]);  add_tensor_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_107: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_525, arg212_1);  view_525 = arg212_1 = None
    add_199: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_107, arg213_1);  mul_107 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_526: "f32[128, 128]" = torch.ops.aten.reshape.default(add_199, [128, 128])
    permute_263: "f32[128, 128]" = torch.ops.aten.permute.default(arg788_1, [1, 0]);  arg788_1 = None
    
    # No stacktrace found for following nodes
    mm_default_164: "f32[128, 128]" = torch.ops.aten.mm.default(view_526, permute_263);  view_526 = permute_263 = None
    add_tensor_164: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_164, arg789_1);  mm_default_164 = arg789_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_527: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_164, [1, 128, 128]);  add_tensor_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_532: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_527, [1, 128, 4, 32]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_266: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    
    # No stacktrace found for following nodes
    clone_default_30: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_528: "f32[128, 128]" = torch.ops.aten.reshape.default(add_199, [128, 128]);  add_199 = None
    permute_264: "f32[128, 128]" = torch.ops.aten.permute.default(arg790_1, [1, 0]);  arg790_1 = None
    
    # No stacktrace found for following nodes
    mm_default_163: "f32[128, 128]" = torch.ops.aten.mm.default(view_528, permute_264);  view_528 = permute_264 = None
    add_tensor_163: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_163, arg791_1);  mm_default_163 = arg791_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_529: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_163, [1, 128, 128]);  add_tensor_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_533: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_529, [1, 128, 4, 32]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_267: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
    
    # No stacktrace found for following nodes
    clone_default_31: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_530: "f32[128, 512]" = torch.ops.aten.reshape.default(add_197, [128, 512])
    permute_265: "f32[512, 128]" = torch.ops.aten.permute.default(arg792_1, [1, 0]);  arg792_1 = None
    
    # No stacktrace found for following nodes
    mm_default_162: "f32[128, 128]" = torch.ops.aten.mm.default(view_530, permute_265);  view_530 = permute_265 = None
    add_tensor_162: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_162, arg793_1);  mm_default_162 = arg793_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_531: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_162, [1, 128, 128]);  add_tensor_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_534: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_531, [1, 128, 4, 32]);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_268: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    
    # No stacktrace found for following nodes
    clone_default_32: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_30, clone_default_31, clone_default_32, None, False, scale = 0.17677669529663687);  clone_default_30 = clone_default_31 = clone_default_32 = None
    getitem_12: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_270: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_541: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_270, [1, 128, 128]);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_542: "f32[128, 128]" = torch.ops.aten.reshape.default(view_541, [128, 128]);  view_541 = None
    permute_271: "f32[128, 128]" = torch.ops.aten.permute.default(arg794_1, [1, 0]);  arg794_1 = None
    
    # No stacktrace found for following nodes
    mm_default_161: "f32[128, 128]" = torch.ops.aten.mm.default(view_542, permute_271);  view_542 = permute_271 = None
    add_tensor_161: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_161, arg795_1);  mm_default_161 = arg795_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_543: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_161, [1, 128, 128]);  add_tensor_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_522: "f32[128, 512]" = torch.ops.aten.reshape.default(add_197, [128, 512])
    permute_261: "f32[512, 128]" = torch.ops.aten.permute.default(arg784_1, [1, 0]);  arg784_1 = None
    
    # No stacktrace found for following nodes
    mm_default_160: "f32[128, 128]" = torch.ops.aten.mm.default(view_522, permute_261);  view_522 = permute_261 = None
    add_tensor_160: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_160, arg785_1);  mm_default_160 = arg785_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_523: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_160, [1, 128, 128]);  add_tensor_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_106: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_523, arg210_1);  view_523 = arg210_1 = None
    add_198: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_106, arg211_1);  mul_106 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_201: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_543, add_198);  view_543 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_108: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_201, arg214_1);  add_201 = arg214_1 = None
    add_202: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_108, arg215_1);  mul_108 = arg215_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_544: "f32[128, 128]" = torch.ops.aten.reshape.default(add_202, [128, 128])
    permute_272: "f32[128, 512]" = torch.ops.aten.permute.default(arg796_1, [1, 0]);  arg796_1 = None
    
    # No stacktrace found for following nodes
    mm_default_159: "f32[128, 512]" = torch.ops.aten.mm.default(view_544, permute_272);  view_544 = permute_272 = None
    add_tensor_159: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_159, arg797_1);  mm_default_159 = arg797_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_545: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_159, [1, 128, 512]);  add_tensor_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_52: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_545);  view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_546: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_52, [128, 512]);  relu_52 = None
    permute_273: "f32[512, 128]" = torch.ops.aten.permute.default(arg798_1, [1, 0]);  arg798_1 = None
    
    # No stacktrace found for following nodes
    mm_default_158: "f32[128, 128]" = torch.ops.aten.mm.default(view_546, permute_273);  view_546 = permute_273 = None
    add_tensor_158: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_158, arg799_1);  mm_default_158 = arg799_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_547: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_158, [1, 128, 128]);  add_tensor_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_203: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_547, add_202);  view_547 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_109: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_203, arg216_1);  add_203 = arg216_1 = None
    add_204: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, arg217_1);  mul_109 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_548: "f32[128, 128]" = torch.ops.aten.reshape.default(add_204, [128, 128])
    permute_274: "f32[128, 512]" = torch.ops.aten.permute.default(arg800_1, [1, 0]);  arg800_1 = None
    
    # No stacktrace found for following nodes
    mm_default_157: "f32[128, 512]" = torch.ops.aten.mm.default(view_548, permute_274);  view_548 = permute_274 = None
    add_tensor_157: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_157, arg801_1);  mm_default_157 = arg801_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_549: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_157, [1, 128, 512]);  add_tensor_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_53: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_549);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_550: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_53, [128, 512]);  relu_53 = None
    permute_275: "f32[512, 128]" = torch.ops.aten.permute.default(arg802_1, [1, 0]);  arg802_1 = None
    
    # No stacktrace found for following nodes
    mm_default_156: "f32[128, 128]" = torch.ops.aten.mm.default(view_550, permute_275);  view_550 = permute_275 = None
    add_tensor_156: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_156, arg803_1);  mm_default_156 = arg803_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_551: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_156, [1, 128, 128]);  add_tensor_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_205: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_551, add_204);  view_551 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_110: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_205, arg218_1);  add_205 = arg218_1 = None
    add_206: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_110, arg219_1);  mul_110 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_552: "f32[128, 128]" = torch.ops.aten.reshape.default(add_206, [128, 128])
    permute_276: "f32[128, 512]" = torch.ops.aten.permute.default(arg804_1, [1, 0]);  arg804_1 = None
    
    # No stacktrace found for following nodes
    mm_default_155: "f32[128, 512]" = torch.ops.aten.mm.default(view_552, permute_276);  view_552 = permute_276 = None
    add_tensor_155: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_155, arg805_1);  mm_default_155 = arg805_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_553: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_155, [1, 128, 512]);  add_tensor_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_54: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_553);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_554: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_54, [128, 512]);  relu_54 = None
    permute_277: "f32[512, 128]" = torch.ops.aten.permute.default(arg806_1, [1, 0]);  arg806_1 = None
    
    # No stacktrace found for following nodes
    mm_default_154: "f32[128, 128]" = torch.ops.aten.mm.default(view_554, permute_277);  view_554 = permute_277 = None
    add_tensor_154: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_154, arg807_1);  mm_default_154 = arg807_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_555: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_154, [1, 128, 128]);  add_tensor_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_207: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_555, add_206);  view_555 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_111: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_207, arg220_1);  add_207 = arg220_1 = None
    add_208: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_111, arg221_1);  mul_111 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_556: "f32[128, 128]" = torch.ops.aten.reshape.default(add_208, [128, 128])
    permute_278: "f32[128, 512]" = torch.ops.aten.permute.default(arg808_1, [1, 0]);  arg808_1 = None
    
    # No stacktrace found for following nodes
    mm_default_153: "f32[128, 512]" = torch.ops.aten.mm.default(view_556, permute_278);  view_556 = permute_278 = None
    add_tensor_153: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_153, arg809_1);  mm_default_153 = arg809_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_557: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_153, [1, 128, 512]);  add_tensor_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_55: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_557);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_558: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_55, [128, 512]);  relu_55 = None
    permute_279: "f32[512, 128]" = torch.ops.aten.permute.default(arg810_1, [1, 0]);  arg810_1 = None
    
    # No stacktrace found for following nodes
    mm_default_152: "f32[128, 128]" = torch.ops.aten.mm.default(view_558, permute_279);  view_558 = permute_279 = None
    add_tensor_152: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_152, arg811_1);  mm_default_152 = arg811_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_559: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_152, [1, 128, 128]);  add_tensor_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_209: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_559, add_208);  view_559 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_112: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_209, arg222_1);  add_209 = arg222_1 = None
    add_210: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_112, arg223_1);  mul_112 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_560: "f32[128, 128]" = torch.ops.aten.reshape.default(add_210, [128, 128]);  add_210 = None
    permute_280: "f32[128, 512]" = torch.ops.aten.permute.default(arg812_1, [1, 0]);  arg812_1 = None
    
    # No stacktrace found for following nodes
    mm_default_151: "f32[128, 512]" = torch.ops.aten.mm.default(view_560, permute_280);  view_560 = permute_280 = None
    add_tensor_151: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_151, arg813_1);  mm_default_151 = arg813_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_561: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_151, [1, 128, 512]);  add_tensor_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_211: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_561, add_197);  view_561 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_113: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_211, arg224_1);  add_211 = arg224_1 = None
    add_212: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_113, arg225_1);  mul_113 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_564: "f32[128, 512]" = torch.ops.aten.reshape.default(add_212, [128, 512])
    permute_282: "f32[512, 128]" = torch.ops.aten.permute.default(arg816_1, [1, 0]);  arg816_1 = None
    
    # No stacktrace found for following nodes
    mm_default_150: "f32[128, 128]" = torch.ops.aten.mm.default(view_564, permute_282);  view_564 = permute_282 = None
    add_tensor_150: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_150, arg817_1);  mm_default_150 = arg817_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_565: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_150, [1, 128, 128]);  add_tensor_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_115: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_565, arg228_1);  view_565 = arg228_1 = None
    add_214: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_115, arg229_1);  mul_115 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_566: "f32[128, 128]" = torch.ops.aten.reshape.default(add_214, [128, 128])
    permute_283: "f32[128, 128]" = torch.ops.aten.permute.default(arg818_1, [1, 0]);  arg818_1 = None
    
    # No stacktrace found for following nodes
    mm_default_149: "f32[128, 128]" = torch.ops.aten.mm.default(view_566, permute_283);  view_566 = permute_283 = None
    add_tensor_149: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_149, arg819_1);  mm_default_149 = arg819_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_567: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_149, [1, 128, 128]);  add_tensor_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_572: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_567, [1, 128, 4, 32]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_286: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
    
    # No stacktrace found for following nodes
    clone_default_27: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_568: "f32[128, 128]" = torch.ops.aten.reshape.default(add_214, [128, 128]);  add_214 = None
    permute_284: "f32[128, 128]" = torch.ops.aten.permute.default(arg820_1, [1, 0]);  arg820_1 = None
    
    # No stacktrace found for following nodes
    mm_default_148: "f32[128, 128]" = torch.ops.aten.mm.default(view_568, permute_284);  view_568 = permute_284 = None
    add_tensor_148: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_148, arg821_1);  mm_default_148 = arg821_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_569: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_148, [1, 128, 128]);  add_tensor_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_573: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_569, [1, 128, 4, 32]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_287: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    
    # No stacktrace found for following nodes
    clone_default_28: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_570: "f32[128, 512]" = torch.ops.aten.reshape.default(add_212, [128, 512])
    permute_285: "f32[512, 128]" = torch.ops.aten.permute.default(arg822_1, [1, 0]);  arg822_1 = None
    
    # No stacktrace found for following nodes
    mm_default_147: "f32[128, 128]" = torch.ops.aten.mm.default(view_570, permute_285);  view_570 = permute_285 = None
    add_tensor_147: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_147, arg823_1);  mm_default_147 = arg823_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_571: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_147, [1, 128, 128]);  add_tensor_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_574: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_571, [1, 128, 4, 32]);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    
    # No stacktrace found for following nodes
    clone_default_29: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_27, clone_default_28, clone_default_29, None, False, scale = 0.17677669529663687);  clone_default_27 = clone_default_28 = clone_default_29 = None
    getitem_11: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_290: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_581: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_290, [1, 128, 128]);  permute_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_582: "f32[128, 128]" = torch.ops.aten.reshape.default(view_581, [128, 128]);  view_581 = None
    permute_291: "f32[128, 128]" = torch.ops.aten.permute.default(arg824_1, [1, 0]);  arg824_1 = None
    
    # No stacktrace found for following nodes
    mm_default_146: "f32[128, 128]" = torch.ops.aten.mm.default(view_582, permute_291);  view_582 = permute_291 = None
    add_tensor_146: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_146, arg825_1);  mm_default_146 = arg825_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_583: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_146, [1, 128, 128]);  add_tensor_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_562: "f32[128, 512]" = torch.ops.aten.reshape.default(add_212, [128, 512])
    permute_281: "f32[512, 128]" = torch.ops.aten.permute.default(arg814_1, [1, 0]);  arg814_1 = None
    
    # No stacktrace found for following nodes
    mm_default_145: "f32[128, 128]" = torch.ops.aten.mm.default(view_562, permute_281);  view_562 = permute_281 = None
    add_tensor_145: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_145, arg815_1);  mm_default_145 = arg815_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_563: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_145, [1, 128, 128]);  add_tensor_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_114: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_563, arg226_1);  view_563 = arg226_1 = None
    add_213: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_114, arg227_1);  mul_114 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_216: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_583, add_213);  view_583 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_116: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_216, arg230_1);  add_216 = arg230_1 = None
    add_217: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_116, arg231_1);  mul_116 = arg231_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_584: "f32[128, 128]" = torch.ops.aten.reshape.default(add_217, [128, 128])
    permute_292: "f32[128, 512]" = torch.ops.aten.permute.default(arg826_1, [1, 0]);  arg826_1 = None
    
    # No stacktrace found for following nodes
    mm_default_144: "f32[128, 512]" = torch.ops.aten.mm.default(view_584, permute_292);  view_584 = permute_292 = None
    add_tensor_144: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_144, arg827_1);  mm_default_144 = arg827_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_585: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_144, [1, 128, 512]);  add_tensor_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_56: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_585);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_586: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_56, [128, 512]);  relu_56 = None
    permute_293: "f32[512, 128]" = torch.ops.aten.permute.default(arg828_1, [1, 0]);  arg828_1 = None
    
    # No stacktrace found for following nodes
    mm_default_143: "f32[128, 128]" = torch.ops.aten.mm.default(view_586, permute_293);  view_586 = permute_293 = None
    add_tensor_143: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_143, arg829_1);  mm_default_143 = arg829_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_587: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_143, [1, 128, 128]);  add_tensor_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_218: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_587, add_217);  view_587 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_117: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_218, arg232_1);  add_218 = arg232_1 = None
    add_219: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_117, arg233_1);  mul_117 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_588: "f32[128, 128]" = torch.ops.aten.reshape.default(add_219, [128, 128])
    permute_294: "f32[128, 512]" = torch.ops.aten.permute.default(arg830_1, [1, 0]);  arg830_1 = None
    
    # No stacktrace found for following nodes
    mm_default_142: "f32[128, 512]" = torch.ops.aten.mm.default(view_588, permute_294);  view_588 = permute_294 = None
    add_tensor_142: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_142, arg831_1);  mm_default_142 = arg831_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_589: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_142, [1, 128, 512]);  add_tensor_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_57: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_589);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_590: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_57, [128, 512]);  relu_57 = None
    permute_295: "f32[512, 128]" = torch.ops.aten.permute.default(arg832_1, [1, 0]);  arg832_1 = None
    
    # No stacktrace found for following nodes
    mm_default_141: "f32[128, 128]" = torch.ops.aten.mm.default(view_590, permute_295);  view_590 = permute_295 = None
    add_tensor_141: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_141, arg833_1);  mm_default_141 = arg833_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_591: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_141, [1, 128, 128]);  add_tensor_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_220: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_591, add_219);  view_591 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_118: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_220, arg234_1);  add_220 = arg234_1 = None
    add_221: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_118, arg235_1);  mul_118 = arg235_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_592: "f32[128, 128]" = torch.ops.aten.reshape.default(add_221, [128, 128])
    permute_296: "f32[128, 512]" = torch.ops.aten.permute.default(arg834_1, [1, 0]);  arg834_1 = None
    
    # No stacktrace found for following nodes
    mm_default_140: "f32[128, 512]" = torch.ops.aten.mm.default(view_592, permute_296);  view_592 = permute_296 = None
    add_tensor_140: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_140, arg835_1);  mm_default_140 = arg835_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_593: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_140, [1, 128, 512]);  add_tensor_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_58: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_593);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_594: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_58, [128, 512]);  relu_58 = None
    permute_297: "f32[512, 128]" = torch.ops.aten.permute.default(arg836_1, [1, 0]);  arg836_1 = None
    
    # No stacktrace found for following nodes
    mm_default_139: "f32[128, 128]" = torch.ops.aten.mm.default(view_594, permute_297);  view_594 = permute_297 = None
    add_tensor_139: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_139, arg837_1);  mm_default_139 = arg837_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_595: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_139, [1, 128, 128]);  add_tensor_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_222: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_595, add_221);  view_595 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_119: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_222, arg236_1);  add_222 = arg236_1 = None
    add_223: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, arg237_1);  mul_119 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_596: "f32[128, 128]" = torch.ops.aten.reshape.default(add_223, [128, 128])
    permute_298: "f32[128, 512]" = torch.ops.aten.permute.default(arg838_1, [1, 0]);  arg838_1 = None
    
    # No stacktrace found for following nodes
    mm_default_138: "f32[128, 512]" = torch.ops.aten.mm.default(view_596, permute_298);  view_596 = permute_298 = None
    add_tensor_138: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_138, arg839_1);  mm_default_138 = arg839_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_597: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_138, [1, 128, 512]);  add_tensor_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_59: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_597);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_598: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_59, [128, 512]);  relu_59 = None
    permute_299: "f32[512, 128]" = torch.ops.aten.permute.default(arg840_1, [1, 0]);  arg840_1 = None
    
    # No stacktrace found for following nodes
    mm_default_137: "f32[128, 128]" = torch.ops.aten.mm.default(view_598, permute_299);  view_598 = permute_299 = None
    add_tensor_137: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_137, arg841_1);  mm_default_137 = arg841_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_599: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_137, [1, 128, 128]);  add_tensor_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_224: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_599, add_223);  view_599 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_120: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_224, arg238_1);  add_224 = arg238_1 = None
    add_225: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_120, arg239_1);  mul_120 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_600: "f32[128, 128]" = torch.ops.aten.reshape.default(add_225, [128, 128]);  add_225 = None
    permute_300: "f32[128, 512]" = torch.ops.aten.permute.default(arg842_1, [1, 0]);  arg842_1 = None
    
    # No stacktrace found for following nodes
    mm_default_136: "f32[128, 512]" = torch.ops.aten.mm.default(view_600, permute_300);  view_600 = permute_300 = None
    add_tensor_136: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_136, arg843_1);  mm_default_136 = arg843_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_601: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_136, [1, 128, 512]);  add_tensor_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_226: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_601, add_212);  view_601 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_121: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_226, arg240_1);  add_226 = arg240_1 = None
    add_227: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_121, arg241_1);  mul_121 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_604: "f32[128, 512]" = torch.ops.aten.reshape.default(add_227, [128, 512])
    permute_302: "f32[512, 128]" = torch.ops.aten.permute.default(arg846_1, [1, 0]);  arg846_1 = None
    
    # No stacktrace found for following nodes
    mm_default_135: "f32[128, 128]" = torch.ops.aten.mm.default(view_604, permute_302);  view_604 = permute_302 = None
    add_tensor_135: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_135, arg847_1);  mm_default_135 = arg847_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_605: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_135, [1, 128, 128]);  add_tensor_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_123: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_605, arg244_1);  view_605 = arg244_1 = None
    add_229: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_123, arg245_1);  mul_123 = arg245_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_606: "f32[128, 128]" = torch.ops.aten.reshape.default(add_229, [128, 128])
    permute_303: "f32[128, 128]" = torch.ops.aten.permute.default(arg848_1, [1, 0]);  arg848_1 = None
    
    # No stacktrace found for following nodes
    mm_default_134: "f32[128, 128]" = torch.ops.aten.mm.default(view_606, permute_303);  view_606 = permute_303 = None
    add_tensor_134: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_134, arg849_1);  mm_default_134 = arg849_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_607: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_134, [1, 128, 128]);  add_tensor_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_612: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_607, [1, 128, 4, 32]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_306: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_608: "f32[128, 128]" = torch.ops.aten.reshape.default(add_229, [128, 128]);  add_229 = None
    permute_304: "f32[128, 128]" = torch.ops.aten.permute.default(arg850_1, [1, 0]);  arg850_1 = None
    
    # No stacktrace found for following nodes
    mm_default_133: "f32[128, 128]" = torch.ops.aten.mm.default(view_608, permute_304);  view_608 = permute_304 = None
    add_tensor_133: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_133, arg851_1);  mm_default_133 = arg851_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_609: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_133, [1, 128, 128]);  add_tensor_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_613: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_609, [1, 128, 4, 32]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_307: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
    
    # No stacktrace found for following nodes
    clone_default_25: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_610: "f32[128, 512]" = torch.ops.aten.reshape.default(add_227, [128, 512])
    permute_305: "f32[512, 128]" = torch.ops.aten.permute.default(arg852_1, [1, 0]);  arg852_1 = None
    
    # No stacktrace found for following nodes
    mm_default_132: "f32[128, 128]" = torch.ops.aten.mm.default(view_610, permute_305);  view_610 = permute_305 = None
    add_tensor_132: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_132, arg853_1);  mm_default_132 = arg853_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_611: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_132, [1, 128, 128]);  add_tensor_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_614: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_611, [1, 128, 4, 32]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_308: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
    
    # No stacktrace found for following nodes
    clone_default_26: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_24, clone_default_25, clone_default_26, None, False, scale = 0.17677669529663687);  clone_default_24 = clone_default_25 = clone_default_26 = None
    getitem_10: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_310: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_621: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_310, [1, 128, 128]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_622: "f32[128, 128]" = torch.ops.aten.reshape.default(view_621, [128, 128]);  view_621 = None
    permute_311: "f32[128, 128]" = torch.ops.aten.permute.default(arg854_1, [1, 0]);  arg854_1 = None
    
    # No stacktrace found for following nodes
    mm_default_131: "f32[128, 128]" = torch.ops.aten.mm.default(view_622, permute_311);  view_622 = permute_311 = None
    add_tensor_131: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_131, arg855_1);  mm_default_131 = arg855_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_623: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_131, [1, 128, 128]);  add_tensor_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_602: "f32[128, 512]" = torch.ops.aten.reshape.default(add_227, [128, 512])
    permute_301: "f32[512, 128]" = torch.ops.aten.permute.default(arg844_1, [1, 0]);  arg844_1 = None
    
    # No stacktrace found for following nodes
    mm_default_130: "f32[128, 128]" = torch.ops.aten.mm.default(view_602, permute_301);  view_602 = permute_301 = None
    add_tensor_130: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_130, arg845_1);  mm_default_130 = arg845_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_603: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_130, [1, 128, 128]);  add_tensor_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_122: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_603, arg242_1);  view_603 = arg242_1 = None
    add_228: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_122, arg243_1);  mul_122 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_231: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_623, add_228);  view_623 = add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_124: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_231, arg246_1);  add_231 = arg246_1 = None
    add_232: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_124, arg247_1);  mul_124 = arg247_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_624: "f32[128, 128]" = torch.ops.aten.reshape.default(add_232, [128, 128])
    permute_312: "f32[128, 512]" = torch.ops.aten.permute.default(arg856_1, [1, 0]);  arg856_1 = None
    
    # No stacktrace found for following nodes
    mm_default_129: "f32[128, 512]" = torch.ops.aten.mm.default(view_624, permute_312);  view_624 = permute_312 = None
    add_tensor_129: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_129, arg857_1);  mm_default_129 = arg857_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_625: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_129, [1, 128, 512]);  add_tensor_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_60: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_625);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_626: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_60, [128, 512]);  relu_60 = None
    permute_313: "f32[512, 128]" = torch.ops.aten.permute.default(arg858_1, [1, 0]);  arg858_1 = None
    
    # No stacktrace found for following nodes
    mm_default_128: "f32[128, 128]" = torch.ops.aten.mm.default(view_626, permute_313);  view_626 = permute_313 = None
    add_tensor_128: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_128, arg859_1);  mm_default_128 = arg859_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_627: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_128, [1, 128, 128]);  add_tensor_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_233: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_627, add_232);  view_627 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_125: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_233, arg248_1);  add_233 = arg248_1 = None
    add_234: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_125, arg249_1);  mul_125 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_628: "f32[128, 128]" = torch.ops.aten.reshape.default(add_234, [128, 128])
    permute_314: "f32[128, 512]" = torch.ops.aten.permute.default(arg860_1, [1, 0]);  arg860_1 = None
    
    # No stacktrace found for following nodes
    mm_default_127: "f32[128, 512]" = torch.ops.aten.mm.default(view_628, permute_314);  view_628 = permute_314 = None
    add_tensor_127: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_127, arg861_1);  mm_default_127 = arg861_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_629: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_127, [1, 128, 512]);  add_tensor_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_61: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_629);  view_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_630: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_61, [128, 512]);  relu_61 = None
    permute_315: "f32[512, 128]" = torch.ops.aten.permute.default(arg862_1, [1, 0]);  arg862_1 = None
    
    # No stacktrace found for following nodes
    mm_default_126: "f32[128, 128]" = torch.ops.aten.mm.default(view_630, permute_315);  view_630 = permute_315 = None
    add_tensor_126: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_126, arg863_1);  mm_default_126 = arg863_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_631: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_126, [1, 128, 128]);  add_tensor_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_235: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_631, add_234);  view_631 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_126: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_235, arg250_1);  add_235 = arg250_1 = None
    add_236: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_126, arg251_1);  mul_126 = arg251_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_632: "f32[128, 128]" = torch.ops.aten.reshape.default(add_236, [128, 128])
    permute_316: "f32[128, 512]" = torch.ops.aten.permute.default(arg864_1, [1, 0]);  arg864_1 = None
    
    # No stacktrace found for following nodes
    mm_default_125: "f32[128, 512]" = torch.ops.aten.mm.default(view_632, permute_316);  view_632 = permute_316 = None
    add_tensor_125: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_125, arg865_1);  mm_default_125 = arg865_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_633: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_125, [1, 128, 512]);  add_tensor_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_62: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_633);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_634: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_62, [128, 512]);  relu_62 = None
    permute_317: "f32[512, 128]" = torch.ops.aten.permute.default(arg866_1, [1, 0]);  arg866_1 = None
    
    # No stacktrace found for following nodes
    mm_default_124: "f32[128, 128]" = torch.ops.aten.mm.default(view_634, permute_317);  view_634 = permute_317 = None
    add_tensor_124: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_124, arg867_1);  mm_default_124 = arg867_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_635: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_124, [1, 128, 128]);  add_tensor_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_237: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_635, add_236);  view_635 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_127: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_237, arg252_1);  add_237 = arg252_1 = None
    add_238: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_127, arg253_1);  mul_127 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_636: "f32[128, 128]" = torch.ops.aten.reshape.default(add_238, [128, 128])
    permute_318: "f32[128, 512]" = torch.ops.aten.permute.default(arg868_1, [1, 0]);  arg868_1 = None
    
    # No stacktrace found for following nodes
    mm_default_123: "f32[128, 512]" = torch.ops.aten.mm.default(view_636, permute_318);  view_636 = permute_318 = None
    add_tensor_123: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_123, arg869_1);  mm_default_123 = arg869_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_637: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_123, [1, 128, 512]);  add_tensor_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_63: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_637);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_638: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_63, [128, 512]);  relu_63 = None
    permute_319: "f32[512, 128]" = torch.ops.aten.permute.default(arg870_1, [1, 0]);  arg870_1 = None
    
    # No stacktrace found for following nodes
    mm_default_122: "f32[128, 128]" = torch.ops.aten.mm.default(view_638, permute_319);  view_638 = permute_319 = None
    add_tensor_122: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_122, arg871_1);  mm_default_122 = arg871_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_639: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_122, [1, 128, 128]);  add_tensor_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_239: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_639, add_238);  view_639 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_128: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_239, arg254_1);  add_239 = arg254_1 = None
    add_240: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, arg255_1);  mul_128 = arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_640: "f32[128, 128]" = torch.ops.aten.reshape.default(add_240, [128, 128]);  add_240 = None
    permute_320: "f32[128, 512]" = torch.ops.aten.permute.default(arg872_1, [1, 0]);  arg872_1 = None
    
    # No stacktrace found for following nodes
    mm_default_121: "f32[128, 512]" = torch.ops.aten.mm.default(view_640, permute_320);  view_640 = permute_320 = None
    add_tensor_121: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_121, arg873_1);  mm_default_121 = arg873_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_641: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_121, [1, 128, 512]);  add_tensor_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_241: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_641, add_227);  view_641 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_129: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_241, arg256_1);  add_241 = arg256_1 = None
    add_242: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_129, arg257_1);  mul_129 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_644: "f32[128, 512]" = torch.ops.aten.reshape.default(add_242, [128, 512])
    permute_322: "f32[512, 128]" = torch.ops.aten.permute.default(arg876_1, [1, 0]);  arg876_1 = None
    
    # No stacktrace found for following nodes
    mm_default_120: "f32[128, 128]" = torch.ops.aten.mm.default(view_644, permute_322);  view_644 = permute_322 = None
    add_tensor_120: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_120, arg877_1);  mm_default_120 = arg877_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_645: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_120, [1, 128, 128]);  add_tensor_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_131: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_645, arg260_1);  view_645 = arg260_1 = None
    add_244: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_131, arg261_1);  mul_131 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_646: "f32[128, 128]" = torch.ops.aten.reshape.default(add_244, [128, 128])
    permute_323: "f32[128, 128]" = torch.ops.aten.permute.default(arg878_1, [1, 0]);  arg878_1 = None
    
    # No stacktrace found for following nodes
    mm_default_119: "f32[128, 128]" = torch.ops.aten.mm.default(view_646, permute_323);  view_646 = permute_323 = None
    add_tensor_119: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_119, arg879_1);  mm_default_119 = arg879_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_647: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_119, [1, 128, 128]);  add_tensor_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_652: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_647, [1, 128, 4, 32]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
    
    # No stacktrace found for following nodes
    clone_default_21: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_648: "f32[128, 128]" = torch.ops.aten.reshape.default(add_244, [128, 128]);  add_244 = None
    permute_324: "f32[128, 128]" = torch.ops.aten.permute.default(arg880_1, [1, 0]);  arg880_1 = None
    
    # No stacktrace found for following nodes
    mm_default_118: "f32[128, 128]" = torch.ops.aten.mm.default(view_648, permute_324);  view_648 = permute_324 = None
    add_tensor_118: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_118, arg881_1);  mm_default_118 = arg881_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_649: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_118, [1, 128, 128]);  add_tensor_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_653: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_649, [1, 128, 4, 32]);  view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    
    # No stacktrace found for following nodes
    clone_default_22: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_650: "f32[128, 512]" = torch.ops.aten.reshape.default(add_242, [128, 512])
    permute_325: "f32[512, 128]" = torch.ops.aten.permute.default(arg882_1, [1, 0]);  arg882_1 = None
    
    # No stacktrace found for following nodes
    mm_default_117: "f32[128, 128]" = torch.ops.aten.mm.default(view_650, permute_325);  view_650 = permute_325 = None
    add_tensor_117: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_117, arg883_1);  mm_default_117 = arg883_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_651: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_117, [1, 128, 128]);  add_tensor_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_654: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_651, [1, 128, 4, 32]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_328: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # No stacktrace found for following nodes
    clone_default_23: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_21, clone_default_22, clone_default_23, None, False, scale = 0.17677669529663687);  clone_default_21 = clone_default_22 = clone_default_23 = None
    getitem_9: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_330: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_661: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_330, [1, 128, 128]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_662: "f32[128, 128]" = torch.ops.aten.reshape.default(view_661, [128, 128]);  view_661 = None
    permute_331: "f32[128, 128]" = torch.ops.aten.permute.default(arg884_1, [1, 0]);  arg884_1 = None
    
    # No stacktrace found for following nodes
    mm_default_116: "f32[128, 128]" = torch.ops.aten.mm.default(view_662, permute_331);  view_662 = permute_331 = None
    add_tensor_116: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_116, arg885_1);  mm_default_116 = arg885_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_663: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_116, [1, 128, 128]);  add_tensor_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_642: "f32[128, 512]" = torch.ops.aten.reshape.default(add_242, [128, 512])
    permute_321: "f32[512, 128]" = torch.ops.aten.permute.default(arg874_1, [1, 0]);  arg874_1 = None
    
    # No stacktrace found for following nodes
    mm_default_115: "f32[128, 128]" = torch.ops.aten.mm.default(view_642, permute_321);  view_642 = permute_321 = None
    add_tensor_115: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_115, arg875_1);  mm_default_115 = arg875_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_643: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_115, [1, 128, 128]);  add_tensor_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_130: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_643, arg258_1);  view_643 = arg258_1 = None
    add_243: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_130, arg259_1);  mul_130 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_246: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_663, add_243);  view_663 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_132: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_246, arg262_1);  add_246 = arg262_1 = None
    add_247: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_132, arg263_1);  mul_132 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_664: "f32[128, 128]" = torch.ops.aten.reshape.default(add_247, [128, 128])
    permute_332: "f32[128, 512]" = torch.ops.aten.permute.default(arg886_1, [1, 0]);  arg886_1 = None
    
    # No stacktrace found for following nodes
    mm_default_114: "f32[128, 512]" = torch.ops.aten.mm.default(view_664, permute_332);  view_664 = permute_332 = None
    add_tensor_114: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_114, arg887_1);  mm_default_114 = arg887_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_665: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_114, [1, 128, 512]);  add_tensor_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_64: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_665);  view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_666: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_64, [128, 512]);  relu_64 = None
    permute_333: "f32[512, 128]" = torch.ops.aten.permute.default(arg888_1, [1, 0]);  arg888_1 = None
    
    # No stacktrace found for following nodes
    mm_default_113: "f32[128, 128]" = torch.ops.aten.mm.default(view_666, permute_333);  view_666 = permute_333 = None
    add_tensor_113: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_113, arg889_1);  mm_default_113 = arg889_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_667: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_113, [1, 128, 128]);  add_tensor_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_248: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_667, add_247);  view_667 = add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_133: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_248, arg264_1);  add_248 = arg264_1 = None
    add_249: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_133, arg265_1);  mul_133 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_668: "f32[128, 128]" = torch.ops.aten.reshape.default(add_249, [128, 128])
    permute_334: "f32[128, 512]" = torch.ops.aten.permute.default(arg890_1, [1, 0]);  arg890_1 = None
    
    # No stacktrace found for following nodes
    mm_default_112: "f32[128, 512]" = torch.ops.aten.mm.default(view_668, permute_334);  view_668 = permute_334 = None
    add_tensor_112: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_112, arg891_1);  mm_default_112 = arg891_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_669: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_112, [1, 128, 512]);  add_tensor_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_65: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_669);  view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_670: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_65, [128, 512]);  relu_65 = None
    permute_335: "f32[512, 128]" = torch.ops.aten.permute.default(arg892_1, [1, 0]);  arg892_1 = None
    
    # No stacktrace found for following nodes
    mm_default_111: "f32[128, 128]" = torch.ops.aten.mm.default(view_670, permute_335);  view_670 = permute_335 = None
    add_tensor_111: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_111, arg893_1);  mm_default_111 = arg893_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_671: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_111, [1, 128, 128]);  add_tensor_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_250: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_671, add_249);  view_671 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_134: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_250, arg266_1);  add_250 = arg266_1 = None
    add_251: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_134, arg267_1);  mul_134 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_672: "f32[128, 128]" = torch.ops.aten.reshape.default(add_251, [128, 128])
    permute_336: "f32[128, 512]" = torch.ops.aten.permute.default(arg894_1, [1, 0]);  arg894_1 = None
    
    # No stacktrace found for following nodes
    mm_default_110: "f32[128, 512]" = torch.ops.aten.mm.default(view_672, permute_336);  view_672 = permute_336 = None
    add_tensor_110: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_110, arg895_1);  mm_default_110 = arg895_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_673: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_110, [1, 128, 512]);  add_tensor_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_66: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_673);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_674: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_66, [128, 512]);  relu_66 = None
    permute_337: "f32[512, 128]" = torch.ops.aten.permute.default(arg896_1, [1, 0]);  arg896_1 = None
    
    # No stacktrace found for following nodes
    mm_default_109: "f32[128, 128]" = torch.ops.aten.mm.default(view_674, permute_337);  view_674 = permute_337 = None
    add_tensor_109: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_109, arg897_1);  mm_default_109 = arg897_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_675: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_109, [1, 128, 128]);  add_tensor_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_252: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_675, add_251);  view_675 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_135: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_252, arg268_1);  add_252 = arg268_1 = None
    add_253: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_135, arg269_1);  mul_135 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_676: "f32[128, 128]" = torch.ops.aten.reshape.default(add_253, [128, 128])
    permute_338: "f32[128, 512]" = torch.ops.aten.permute.default(arg898_1, [1, 0]);  arg898_1 = None
    
    # No stacktrace found for following nodes
    mm_default_108: "f32[128, 512]" = torch.ops.aten.mm.default(view_676, permute_338);  view_676 = permute_338 = None
    add_tensor_108: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_108, arg899_1);  mm_default_108 = arg899_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_677: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_108, [1, 128, 512]);  add_tensor_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_67: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_677);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_678: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_67, [128, 512]);  relu_67 = None
    permute_339: "f32[512, 128]" = torch.ops.aten.permute.default(arg900_1, [1, 0]);  arg900_1 = None
    
    # No stacktrace found for following nodes
    mm_default_107: "f32[128, 128]" = torch.ops.aten.mm.default(view_678, permute_339);  view_678 = permute_339 = None
    add_tensor_107: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_107, arg901_1);  mm_default_107 = arg901_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_679: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_107, [1, 128, 128]);  add_tensor_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_254: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_679, add_253);  view_679 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_136: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_254, arg270_1);  add_254 = arg270_1 = None
    add_255: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_136, arg271_1);  mul_136 = arg271_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_680: "f32[128, 128]" = torch.ops.aten.reshape.default(add_255, [128, 128]);  add_255 = None
    permute_340: "f32[128, 512]" = torch.ops.aten.permute.default(arg902_1, [1, 0]);  arg902_1 = None
    
    # No stacktrace found for following nodes
    mm_default_106: "f32[128, 512]" = torch.ops.aten.mm.default(view_680, permute_340);  view_680 = permute_340 = None
    add_tensor_106: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_106, arg903_1);  mm_default_106 = arg903_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_681: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_106, [1, 128, 512]);  add_tensor_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_256: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_681, add_242);  view_681 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_137: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_256, arg272_1);  add_256 = arg272_1 = None
    add_257: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_137, arg273_1);  mul_137 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_684: "f32[128, 512]" = torch.ops.aten.reshape.default(add_257, [128, 512])
    permute_342: "f32[512, 128]" = torch.ops.aten.permute.default(arg906_1, [1, 0]);  arg906_1 = None
    
    # No stacktrace found for following nodes
    mm_default_105: "f32[128, 128]" = torch.ops.aten.mm.default(view_684, permute_342);  view_684 = permute_342 = None
    add_tensor_105: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_105, arg907_1);  mm_default_105 = arg907_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_685: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_105, [1, 128, 128]);  add_tensor_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_139: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_685, arg276_1);  view_685 = arg276_1 = None
    add_259: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_139, arg277_1);  mul_139 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_686: "f32[128, 128]" = torch.ops.aten.reshape.default(add_259, [128, 128])
    permute_343: "f32[128, 128]" = torch.ops.aten.permute.default(arg908_1, [1, 0]);  arg908_1 = None
    
    # No stacktrace found for following nodes
    mm_default_104: "f32[128, 128]" = torch.ops.aten.mm.default(view_686, permute_343);  view_686 = permute_343 = None
    add_tensor_104: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_104, arg909_1);  mm_default_104 = arg909_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_687: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_104, [1, 128, 128]);  add_tensor_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_692: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_687, [1, 128, 4, 32]);  view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_346: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
    
    # No stacktrace found for following nodes
    clone_default_18: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_688: "f32[128, 128]" = torch.ops.aten.reshape.default(add_259, [128, 128]);  add_259 = None
    permute_344: "f32[128, 128]" = torch.ops.aten.permute.default(arg910_1, [1, 0]);  arg910_1 = None
    
    # No stacktrace found for following nodes
    mm_default_103: "f32[128, 128]" = torch.ops.aten.mm.default(view_688, permute_344);  view_688 = permute_344 = None
    add_tensor_103: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_103, arg911_1);  mm_default_103 = arg911_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_689: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_103, [1, 128, 128]);  add_tensor_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_693: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_689, [1, 128, 4, 32]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_347: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
    
    # No stacktrace found for following nodes
    clone_default_19: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_690: "f32[128, 512]" = torch.ops.aten.reshape.default(add_257, [128, 512])
    permute_345: "f32[512, 128]" = torch.ops.aten.permute.default(arg912_1, [1, 0]);  arg912_1 = None
    
    # No stacktrace found for following nodes
    mm_default_102: "f32[128, 128]" = torch.ops.aten.mm.default(view_690, permute_345);  view_690 = permute_345 = None
    add_tensor_102: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_102, arg913_1);  mm_default_102 = arg913_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_691: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_102, [1, 128, 128]);  add_tensor_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_694: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_691, [1, 128, 4, 32]);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_348: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
    
    # No stacktrace found for following nodes
    clone_default_20: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_18, clone_default_19, clone_default_20, None, False, scale = 0.17677669529663687);  clone_default_18 = clone_default_19 = clone_default_20 = None
    getitem_8: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_350: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_701: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_350, [1, 128, 128]);  permute_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_702: "f32[128, 128]" = torch.ops.aten.reshape.default(view_701, [128, 128]);  view_701 = None
    permute_351: "f32[128, 128]" = torch.ops.aten.permute.default(arg914_1, [1, 0]);  arg914_1 = None
    
    # No stacktrace found for following nodes
    mm_default_101: "f32[128, 128]" = torch.ops.aten.mm.default(view_702, permute_351);  view_702 = permute_351 = None
    add_tensor_101: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_101, arg915_1);  mm_default_101 = arg915_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_703: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_101, [1, 128, 128]);  add_tensor_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_682: "f32[128, 512]" = torch.ops.aten.reshape.default(add_257, [128, 512])
    permute_341: "f32[512, 128]" = torch.ops.aten.permute.default(arg904_1, [1, 0]);  arg904_1 = None
    
    # No stacktrace found for following nodes
    mm_default_100: "f32[128, 128]" = torch.ops.aten.mm.default(view_682, permute_341);  view_682 = permute_341 = None
    add_tensor_100: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_100, arg905_1);  mm_default_100 = arg905_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_683: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_100, [1, 128, 128]);  add_tensor_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_138: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_683, arg274_1);  view_683 = arg274_1 = None
    add_258: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_138, arg275_1);  mul_138 = arg275_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_261: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_703, add_258);  view_703 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_140: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_261, arg278_1);  add_261 = arg278_1 = None
    add_262: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_140, arg279_1);  mul_140 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_704: "f32[128, 128]" = torch.ops.aten.reshape.default(add_262, [128, 128])
    permute_352: "f32[128, 512]" = torch.ops.aten.permute.default(arg916_1, [1, 0]);  arg916_1 = None
    
    # No stacktrace found for following nodes
    mm_default_99: "f32[128, 512]" = torch.ops.aten.mm.default(view_704, permute_352);  view_704 = permute_352 = None
    add_tensor_99: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_99, arg917_1);  mm_default_99 = arg917_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_705: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_99, [1, 128, 512]);  add_tensor_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_68: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_705);  view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_706: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_68, [128, 512]);  relu_68 = None
    permute_353: "f32[512, 128]" = torch.ops.aten.permute.default(arg918_1, [1, 0]);  arg918_1 = None
    
    # No stacktrace found for following nodes
    mm_default_98: "f32[128, 128]" = torch.ops.aten.mm.default(view_706, permute_353);  view_706 = permute_353 = None
    add_tensor_98: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_98, arg919_1);  mm_default_98 = arg919_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_707: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_98, [1, 128, 128]);  add_tensor_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_263: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_707, add_262);  view_707 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_141: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_263, arg280_1);  add_263 = arg280_1 = None
    add_264: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_141, arg281_1);  mul_141 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_708: "f32[128, 128]" = torch.ops.aten.reshape.default(add_264, [128, 128])
    permute_354: "f32[128, 512]" = torch.ops.aten.permute.default(arg920_1, [1, 0]);  arg920_1 = None
    
    # No stacktrace found for following nodes
    mm_default_97: "f32[128, 512]" = torch.ops.aten.mm.default(view_708, permute_354);  view_708 = permute_354 = None
    add_tensor_97: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_97, arg921_1);  mm_default_97 = arg921_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_709: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_97, [1, 128, 512]);  add_tensor_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_69: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_709);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_710: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_69, [128, 512]);  relu_69 = None
    permute_355: "f32[512, 128]" = torch.ops.aten.permute.default(arg922_1, [1, 0]);  arg922_1 = None
    
    # No stacktrace found for following nodes
    mm_default_96: "f32[128, 128]" = torch.ops.aten.mm.default(view_710, permute_355);  view_710 = permute_355 = None
    add_tensor_96: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_96, arg923_1);  mm_default_96 = arg923_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_711: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_96, [1, 128, 128]);  add_tensor_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_265: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_711, add_264);  view_711 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_142: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_265, arg282_1);  add_265 = arg282_1 = None
    add_266: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_142, arg283_1);  mul_142 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_712: "f32[128, 128]" = torch.ops.aten.reshape.default(add_266, [128, 128])
    permute_356: "f32[128, 512]" = torch.ops.aten.permute.default(arg924_1, [1, 0]);  arg924_1 = None
    
    # No stacktrace found for following nodes
    mm_default_95: "f32[128, 512]" = torch.ops.aten.mm.default(view_712, permute_356);  view_712 = permute_356 = None
    add_tensor_95: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_95, arg925_1);  mm_default_95 = arg925_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_713: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_95, [1, 128, 512]);  add_tensor_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_70: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_713);  view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_714: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_70, [128, 512]);  relu_70 = None
    permute_357: "f32[512, 128]" = torch.ops.aten.permute.default(arg926_1, [1, 0]);  arg926_1 = None
    
    # No stacktrace found for following nodes
    mm_default_94: "f32[128, 128]" = torch.ops.aten.mm.default(view_714, permute_357);  view_714 = permute_357 = None
    add_tensor_94: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_94, arg927_1);  mm_default_94 = arg927_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_715: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_94, [1, 128, 128]);  add_tensor_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_267: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_715, add_266);  view_715 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_143: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_267, arg284_1);  add_267 = arg284_1 = None
    add_268: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_143, arg285_1);  mul_143 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_716: "f32[128, 128]" = torch.ops.aten.reshape.default(add_268, [128, 128])
    permute_358: "f32[128, 512]" = torch.ops.aten.permute.default(arg928_1, [1, 0]);  arg928_1 = None
    
    # No stacktrace found for following nodes
    mm_default_93: "f32[128, 512]" = torch.ops.aten.mm.default(view_716, permute_358);  view_716 = permute_358 = None
    add_tensor_93: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_93, arg929_1);  mm_default_93 = arg929_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_717: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_93, [1, 128, 512]);  add_tensor_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_71: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_717);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_718: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_71, [128, 512]);  relu_71 = None
    permute_359: "f32[512, 128]" = torch.ops.aten.permute.default(arg930_1, [1, 0]);  arg930_1 = None
    
    # No stacktrace found for following nodes
    mm_default_92: "f32[128, 128]" = torch.ops.aten.mm.default(view_718, permute_359);  view_718 = permute_359 = None
    add_tensor_92: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_92, arg931_1);  mm_default_92 = arg931_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_719: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_92, [1, 128, 128]);  add_tensor_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_269: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_719, add_268);  view_719 = add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_144: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_269, arg286_1);  add_269 = arg286_1 = None
    add_270: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_144, arg287_1);  mul_144 = arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_720: "f32[128, 128]" = torch.ops.aten.reshape.default(add_270, [128, 128]);  add_270 = None
    permute_360: "f32[128, 512]" = torch.ops.aten.permute.default(arg932_1, [1, 0]);  arg932_1 = None
    
    # No stacktrace found for following nodes
    mm_default_91: "f32[128, 512]" = torch.ops.aten.mm.default(view_720, permute_360);  view_720 = permute_360 = None
    add_tensor_91: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_91, arg933_1);  mm_default_91 = arg933_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_721: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_91, [1, 128, 512]);  add_tensor_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_271: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_721, add_257);  view_721 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_145: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_271, arg288_1);  add_271 = arg288_1 = None
    add_272: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_145, arg289_1);  mul_145 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_724: "f32[128, 512]" = torch.ops.aten.reshape.default(add_272, [128, 512])
    permute_362: "f32[512, 128]" = torch.ops.aten.permute.default(arg936_1, [1, 0]);  arg936_1 = None
    
    # No stacktrace found for following nodes
    mm_default_90: "f32[128, 128]" = torch.ops.aten.mm.default(view_724, permute_362);  view_724 = permute_362 = None
    add_tensor_90: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_90, arg937_1);  mm_default_90 = arg937_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_725: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_90, [1, 128, 128]);  add_tensor_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_147: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_725, arg292_1);  view_725 = arg292_1 = None
    add_274: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_147, arg293_1);  mul_147 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_726: "f32[128, 128]" = torch.ops.aten.reshape.default(add_274, [128, 128])
    permute_363: "f32[128, 128]" = torch.ops.aten.permute.default(arg938_1, [1, 0]);  arg938_1 = None
    
    # No stacktrace found for following nodes
    mm_default_89: "f32[128, 128]" = torch.ops.aten.mm.default(view_726, permute_363);  view_726 = permute_363 = None
    add_tensor_89: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_89, arg939_1);  mm_default_89 = arg939_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_727: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_89, [1, 128, 128]);  add_tensor_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_732: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_727, [1, 128, 4, 32]);  view_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_366: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
    
    # No stacktrace found for following nodes
    clone_default_15: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_728: "f32[128, 128]" = torch.ops.aten.reshape.default(add_274, [128, 128]);  add_274 = None
    permute_364: "f32[128, 128]" = torch.ops.aten.permute.default(arg940_1, [1, 0]);  arg940_1 = None
    
    # No stacktrace found for following nodes
    mm_default_88: "f32[128, 128]" = torch.ops.aten.mm.default(view_728, permute_364);  view_728 = permute_364 = None
    add_tensor_88: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_88, arg941_1);  mm_default_88 = arg941_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_729: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_88, [1, 128, 128]);  add_tensor_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_733: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_729, [1, 128, 4, 32]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_367: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_733, [0, 2, 1, 3]);  view_733 = None
    
    # No stacktrace found for following nodes
    clone_default_16: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_730: "f32[128, 512]" = torch.ops.aten.reshape.default(add_272, [128, 512])
    permute_365: "f32[512, 128]" = torch.ops.aten.permute.default(arg942_1, [1, 0]);  arg942_1 = None
    
    # No stacktrace found for following nodes
    mm_default_87: "f32[128, 128]" = torch.ops.aten.mm.default(view_730, permute_365);  view_730 = permute_365 = None
    add_tensor_87: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_87, arg943_1);  mm_default_87 = arg943_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_731: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_87, [1, 128, 128]);  add_tensor_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_734: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_731, [1, 128, 4, 32]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_368: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
    
    # No stacktrace found for following nodes
    clone_default_17: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_15, clone_default_16, clone_default_17, None, False, scale = 0.17677669529663687);  clone_default_15 = clone_default_16 = clone_default_17 = None
    getitem_7: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_370: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_7, [0, 2, 1, 3]);  getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_741: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_370, [1, 128, 128]);  permute_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_742: "f32[128, 128]" = torch.ops.aten.reshape.default(view_741, [128, 128]);  view_741 = None
    permute_371: "f32[128, 128]" = torch.ops.aten.permute.default(arg944_1, [1, 0]);  arg944_1 = None
    
    # No stacktrace found for following nodes
    mm_default_86: "f32[128, 128]" = torch.ops.aten.mm.default(view_742, permute_371);  view_742 = permute_371 = None
    add_tensor_86: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_86, arg945_1);  mm_default_86 = arg945_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_743: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_86, [1, 128, 128]);  add_tensor_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_722: "f32[128, 512]" = torch.ops.aten.reshape.default(add_272, [128, 512])
    permute_361: "f32[512, 128]" = torch.ops.aten.permute.default(arg934_1, [1, 0]);  arg934_1 = None
    
    # No stacktrace found for following nodes
    mm_default_85: "f32[128, 128]" = torch.ops.aten.mm.default(view_722, permute_361);  view_722 = permute_361 = None
    add_tensor_85: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_85, arg935_1);  mm_default_85 = arg935_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_723: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_85, [1, 128, 128]);  add_tensor_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_146: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_723, arg290_1);  view_723 = arg290_1 = None
    add_273: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_146, arg291_1);  mul_146 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_276: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_743, add_273);  view_743 = add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_148: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_276, arg294_1);  add_276 = arg294_1 = None
    add_277: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_148, arg295_1);  mul_148 = arg295_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_744: "f32[128, 128]" = torch.ops.aten.reshape.default(add_277, [128, 128])
    permute_372: "f32[128, 512]" = torch.ops.aten.permute.default(arg946_1, [1, 0]);  arg946_1 = None
    
    # No stacktrace found for following nodes
    mm_default_84: "f32[128, 512]" = torch.ops.aten.mm.default(view_744, permute_372);  view_744 = permute_372 = None
    add_tensor_84: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_84, arg947_1);  mm_default_84 = arg947_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_745: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_84, [1, 128, 512]);  add_tensor_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_72: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_745);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_746: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_72, [128, 512]);  relu_72 = None
    permute_373: "f32[512, 128]" = torch.ops.aten.permute.default(arg948_1, [1, 0]);  arg948_1 = None
    
    # No stacktrace found for following nodes
    mm_default_83: "f32[128, 128]" = torch.ops.aten.mm.default(view_746, permute_373);  view_746 = permute_373 = None
    add_tensor_83: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_83, arg949_1);  mm_default_83 = arg949_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_747: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_83, [1, 128, 128]);  add_tensor_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_278: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_747, add_277);  view_747 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_149: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_278, arg296_1);  add_278 = arg296_1 = None
    add_279: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_149, arg297_1);  mul_149 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_748: "f32[128, 128]" = torch.ops.aten.reshape.default(add_279, [128, 128])
    permute_374: "f32[128, 512]" = torch.ops.aten.permute.default(arg950_1, [1, 0]);  arg950_1 = None
    
    # No stacktrace found for following nodes
    mm_default_82: "f32[128, 512]" = torch.ops.aten.mm.default(view_748, permute_374);  view_748 = permute_374 = None
    add_tensor_82: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_82, arg951_1);  mm_default_82 = arg951_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_749: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_82, [1, 128, 512]);  add_tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_73: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_749);  view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_750: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_73, [128, 512]);  relu_73 = None
    permute_375: "f32[512, 128]" = torch.ops.aten.permute.default(arg952_1, [1, 0]);  arg952_1 = None
    
    # No stacktrace found for following nodes
    mm_default_81: "f32[128, 128]" = torch.ops.aten.mm.default(view_750, permute_375);  view_750 = permute_375 = None
    add_tensor_81: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_81, arg953_1);  mm_default_81 = arg953_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_751: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_81, [1, 128, 128]);  add_tensor_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_280: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_751, add_279);  view_751 = add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_150: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_280, arg298_1);  add_280 = arg298_1 = None
    add_281: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_150, arg299_1);  mul_150 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_752: "f32[128, 128]" = torch.ops.aten.reshape.default(add_281, [128, 128])
    permute_376: "f32[128, 512]" = torch.ops.aten.permute.default(arg954_1, [1, 0]);  arg954_1 = None
    
    # No stacktrace found for following nodes
    mm_default_80: "f32[128, 512]" = torch.ops.aten.mm.default(view_752, permute_376);  view_752 = permute_376 = None
    add_tensor_80: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_80, arg955_1);  mm_default_80 = arg955_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_753: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_80, [1, 128, 512]);  add_tensor_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_74: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_753);  view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_754: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_74, [128, 512]);  relu_74 = None
    permute_377: "f32[512, 128]" = torch.ops.aten.permute.default(arg956_1, [1, 0]);  arg956_1 = None
    
    # No stacktrace found for following nodes
    mm_default_79: "f32[128, 128]" = torch.ops.aten.mm.default(view_754, permute_377);  view_754 = permute_377 = None
    add_tensor_79: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_79, arg957_1);  mm_default_79 = arg957_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_755: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_79, [1, 128, 128]);  add_tensor_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_282: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_755, add_281);  view_755 = add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_151: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_282, arg300_1);  add_282 = arg300_1 = None
    add_283: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_151, arg301_1);  mul_151 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_756: "f32[128, 128]" = torch.ops.aten.reshape.default(add_283, [128, 128])
    permute_378: "f32[128, 512]" = torch.ops.aten.permute.default(arg958_1, [1, 0]);  arg958_1 = None
    
    # No stacktrace found for following nodes
    mm_default_78: "f32[128, 512]" = torch.ops.aten.mm.default(view_756, permute_378);  view_756 = permute_378 = None
    add_tensor_78: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_78, arg959_1);  mm_default_78 = arg959_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_757: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_78, [1, 128, 512]);  add_tensor_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_75: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_757);  view_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_758: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_75, [128, 512]);  relu_75 = None
    permute_379: "f32[512, 128]" = torch.ops.aten.permute.default(arg960_1, [1, 0]);  arg960_1 = None
    
    # No stacktrace found for following nodes
    mm_default_77: "f32[128, 128]" = torch.ops.aten.mm.default(view_758, permute_379);  view_758 = permute_379 = None
    add_tensor_77: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_77, arg961_1);  mm_default_77 = arg961_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_759: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_77, [1, 128, 128]);  add_tensor_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_284: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_759, add_283);  view_759 = add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_152: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_284, arg302_1);  add_284 = arg302_1 = None
    add_285: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_152, arg303_1);  mul_152 = arg303_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_760: "f32[128, 128]" = torch.ops.aten.reshape.default(add_285, [128, 128]);  add_285 = None
    permute_380: "f32[128, 512]" = torch.ops.aten.permute.default(arg962_1, [1, 0]);  arg962_1 = None
    
    # No stacktrace found for following nodes
    mm_default_76: "f32[128, 512]" = torch.ops.aten.mm.default(view_760, permute_380);  view_760 = permute_380 = None
    add_tensor_76: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_76, arg963_1);  mm_default_76 = arg963_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_761: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_76, [1, 128, 512]);  add_tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_286: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_761, add_272);  view_761 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_153: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_286, arg304_1);  add_286 = arg304_1 = None
    add_287: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_153, arg305_1);  mul_153 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_764: "f32[128, 512]" = torch.ops.aten.reshape.default(add_287, [128, 512])
    permute_382: "f32[512, 128]" = torch.ops.aten.permute.default(arg966_1, [1, 0]);  arg966_1 = None
    
    # No stacktrace found for following nodes
    mm_default_75: "f32[128, 128]" = torch.ops.aten.mm.default(view_764, permute_382);  view_764 = permute_382 = None
    add_tensor_75: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_75, arg967_1);  mm_default_75 = arg967_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_765: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_75, [1, 128, 128]);  add_tensor_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_155: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_765, arg308_1);  view_765 = arg308_1 = None
    add_289: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_155, arg309_1);  mul_155 = arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_766: "f32[128, 128]" = torch.ops.aten.reshape.default(add_289, [128, 128])
    permute_383: "f32[128, 128]" = torch.ops.aten.permute.default(arg968_1, [1, 0]);  arg968_1 = None
    
    # No stacktrace found for following nodes
    mm_default_74: "f32[128, 128]" = torch.ops.aten.mm.default(view_766, permute_383);  view_766 = permute_383 = None
    add_tensor_74: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_74, arg969_1);  mm_default_74 = arg969_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_767: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_74, [1, 128, 128]);  add_tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_772: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_767, [1, 128, 4, 32]);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_386: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_768: "f32[128, 128]" = torch.ops.aten.reshape.default(add_289, [128, 128]);  add_289 = None
    permute_384: "f32[128, 128]" = torch.ops.aten.permute.default(arg970_1, [1, 0]);  arg970_1 = None
    
    # No stacktrace found for following nodes
    mm_default_73: "f32[128, 128]" = torch.ops.aten.mm.default(view_768, permute_384);  view_768 = permute_384 = None
    add_tensor_73: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_73, arg971_1);  mm_default_73 = arg971_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_769: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_73, [1, 128, 128]);  add_tensor_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_773: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_769, [1, 128, 4, 32]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None
    
    # No stacktrace found for following nodes
    clone_default_13: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_770: "f32[128, 512]" = torch.ops.aten.reshape.default(add_287, [128, 512])
    permute_385: "f32[512, 128]" = torch.ops.aten.permute.default(arg972_1, [1, 0]);  arg972_1 = None
    
    # No stacktrace found for following nodes
    mm_default_72: "f32[128, 128]" = torch.ops.aten.mm.default(view_770, permute_385);  view_770 = permute_385 = None
    add_tensor_72: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_72, arg973_1);  mm_default_72 = arg973_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_771: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_72, [1, 128, 128]);  add_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_774: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_771, [1, 128, 4, 32]);  view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_388: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
    
    # No stacktrace found for following nodes
    clone_default_14: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_12, clone_default_13, clone_default_14, None, False, scale = 0.17677669529663687);  clone_default_12 = clone_default_13 = clone_default_14 = None
    getitem_6: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_390: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_781: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_390, [1, 128, 128]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_782: "f32[128, 128]" = torch.ops.aten.reshape.default(view_781, [128, 128]);  view_781 = None
    permute_391: "f32[128, 128]" = torch.ops.aten.permute.default(arg974_1, [1, 0]);  arg974_1 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[128, 128]" = torch.ops.aten.mm.default(view_782, permute_391);  view_782 = permute_391 = None
    add_tensor_71: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_71, arg975_1);  mm_default_71 = arg975_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_783: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_71, [1, 128, 128]);  add_tensor_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_762: "f32[128, 512]" = torch.ops.aten.reshape.default(add_287, [128, 512])
    permute_381: "f32[512, 128]" = torch.ops.aten.permute.default(arg964_1, [1, 0]);  arg964_1 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[128, 128]" = torch.ops.aten.mm.default(view_762, permute_381);  view_762 = permute_381 = None
    add_tensor_70: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_70, arg965_1);  mm_default_70 = arg965_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_763: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_70, [1, 128, 128]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_154: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_763, arg306_1);  view_763 = arg306_1 = None
    add_288: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_154, arg307_1);  mul_154 = arg307_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_291: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_783, add_288);  view_783 = add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_156: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_291, arg310_1);  add_291 = arg310_1 = None
    add_292: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_156, arg311_1);  mul_156 = arg311_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_784: "f32[128, 128]" = torch.ops.aten.reshape.default(add_292, [128, 128])
    permute_392: "f32[128, 512]" = torch.ops.aten.permute.default(arg976_1, [1, 0]);  arg976_1 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[128, 512]" = torch.ops.aten.mm.default(view_784, permute_392);  view_784 = permute_392 = None
    add_tensor_69: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_69, arg977_1);  mm_default_69 = arg977_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_785: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_69, [1, 128, 512]);  add_tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_76: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_785);  view_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_786: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_76, [128, 512]);  relu_76 = None
    permute_393: "f32[512, 128]" = torch.ops.aten.permute.default(arg978_1, [1, 0]);  arg978_1 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[128, 128]" = torch.ops.aten.mm.default(view_786, permute_393);  view_786 = permute_393 = None
    add_tensor_68: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_68, arg979_1);  mm_default_68 = arg979_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_787: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_68, [1, 128, 128]);  add_tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_293: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_787, add_292);  view_787 = add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_157: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, arg312_1);  add_293 = arg312_1 = None
    add_294: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_157, arg313_1);  mul_157 = arg313_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_788: "f32[128, 128]" = torch.ops.aten.reshape.default(add_294, [128, 128])
    permute_394: "f32[128, 512]" = torch.ops.aten.permute.default(arg980_1, [1, 0]);  arg980_1 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[128, 512]" = torch.ops.aten.mm.default(view_788, permute_394);  view_788 = permute_394 = None
    add_tensor_67: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_67, arg981_1);  mm_default_67 = arg981_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_789: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_67, [1, 128, 512]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_77: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_789);  view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_790: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_77, [128, 512]);  relu_77 = None
    permute_395: "f32[512, 128]" = torch.ops.aten.permute.default(arg982_1, [1, 0]);  arg982_1 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[128, 128]" = torch.ops.aten.mm.default(view_790, permute_395);  view_790 = permute_395 = None
    add_tensor_66: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_66, arg983_1);  mm_default_66 = arg983_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_791: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_66, [1, 128, 128]);  add_tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_295: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_791, add_294);  view_791 = add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_158: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_295, arg314_1);  add_295 = arg314_1 = None
    add_296: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_158, arg315_1);  mul_158 = arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_792: "f32[128, 128]" = torch.ops.aten.reshape.default(add_296, [128, 128])
    permute_396: "f32[128, 512]" = torch.ops.aten.permute.default(arg984_1, [1, 0]);  arg984_1 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[128, 512]" = torch.ops.aten.mm.default(view_792, permute_396);  view_792 = permute_396 = None
    add_tensor_65: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_65, arg985_1);  mm_default_65 = arg985_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_793: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_65, [1, 128, 512]);  add_tensor_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_78: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_793);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_794: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_78, [128, 512]);  relu_78 = None
    permute_397: "f32[512, 128]" = torch.ops.aten.permute.default(arg986_1, [1, 0]);  arg986_1 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[128, 128]" = torch.ops.aten.mm.default(view_794, permute_397);  view_794 = permute_397 = None
    add_tensor_64: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_64, arg987_1);  mm_default_64 = arg987_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_795: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_64, [1, 128, 128]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_297: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_795, add_296);  view_795 = add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_159: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_297, arg316_1);  add_297 = arg316_1 = None
    add_298: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_159, arg317_1);  mul_159 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_796: "f32[128, 128]" = torch.ops.aten.reshape.default(add_298, [128, 128])
    permute_398: "f32[128, 512]" = torch.ops.aten.permute.default(arg988_1, [1, 0]);  arg988_1 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[128, 512]" = torch.ops.aten.mm.default(view_796, permute_398);  view_796 = permute_398 = None
    add_tensor_63: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_63, arg989_1);  mm_default_63 = arg989_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_797: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_63, [1, 128, 512]);  add_tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_79: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_797);  view_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_798: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_79, [128, 512]);  relu_79 = None
    permute_399: "f32[512, 128]" = torch.ops.aten.permute.default(arg990_1, [1, 0]);  arg990_1 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[128, 128]" = torch.ops.aten.mm.default(view_798, permute_399);  view_798 = permute_399 = None
    add_tensor_62: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_62, arg991_1);  mm_default_62 = arg991_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_799: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_62, [1, 128, 128]);  add_tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_299: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_799, add_298);  view_799 = add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_160: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_299, arg318_1);  add_299 = arg318_1 = None
    add_300: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_160, arg319_1);  mul_160 = arg319_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_800: "f32[128, 128]" = torch.ops.aten.reshape.default(add_300, [128, 128]);  add_300 = None
    permute_400: "f32[128, 512]" = torch.ops.aten.permute.default(arg992_1, [1, 0]);  arg992_1 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[128, 512]" = torch.ops.aten.mm.default(view_800, permute_400);  view_800 = permute_400 = None
    add_tensor_61: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_61, arg993_1);  mm_default_61 = arg993_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_801: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_61, [1, 128, 512]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_301: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_801, add_287);  view_801 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_161: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_301, arg320_1);  add_301 = arg320_1 = None
    add_302: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_161, arg321_1);  mul_161 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_804: "f32[128, 512]" = torch.ops.aten.reshape.default(add_302, [128, 512])
    permute_402: "f32[512, 128]" = torch.ops.aten.permute.default(arg996_1, [1, 0]);  arg996_1 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[128, 128]" = torch.ops.aten.mm.default(view_804, permute_402);  view_804 = permute_402 = None
    add_tensor_60: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_60, arg997_1);  mm_default_60 = arg997_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_805: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_60, [1, 128, 128]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_163: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_805, arg324_1);  view_805 = arg324_1 = None
    add_304: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_163, arg325_1);  mul_163 = arg325_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_806: "f32[128, 128]" = torch.ops.aten.reshape.default(add_304, [128, 128])
    permute_403: "f32[128, 128]" = torch.ops.aten.permute.default(arg998_1, [1, 0]);  arg998_1 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[128, 128]" = torch.ops.aten.mm.default(view_806, permute_403);  view_806 = permute_403 = None
    add_tensor_59: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_59, arg999_1);  mm_default_59 = arg999_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_807: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_59, [1, 128, 128]);  add_tensor_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_812: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_807, [1, 128, 4, 32]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_406: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_812, [0, 2, 1, 3]);  view_812 = None
    
    # No stacktrace found for following nodes
    clone_default_9: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_406, memory_format = torch.contiguous_format);  permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_808: "f32[128, 128]" = torch.ops.aten.reshape.default(add_304, [128, 128]);  add_304 = None
    permute_404: "f32[128, 128]" = torch.ops.aten.permute.default(arg1000_1, [1, 0]);  arg1000_1 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[128, 128]" = torch.ops.aten.mm.default(view_808, permute_404);  view_808 = permute_404 = None
    add_tensor_58: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_58, arg1001_1);  mm_default_58 = arg1001_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_809: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_58, [1, 128, 128]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_813: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_809, [1, 128, 4, 32]);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_407: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_813, [0, 2, 1, 3]);  view_813 = None
    
    # No stacktrace found for following nodes
    clone_default_10: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_810: "f32[128, 512]" = torch.ops.aten.reshape.default(add_302, [128, 512])
    permute_405: "f32[512, 128]" = torch.ops.aten.permute.default(arg1002_1, [1, 0]);  arg1002_1 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[128, 128]" = torch.ops.aten.mm.default(view_810, permute_405);  view_810 = permute_405 = None
    add_tensor_57: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_57, arg1003_1);  mm_default_57 = arg1003_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_811: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_57, [1, 128, 128]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_814: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_811, [1, 128, 4, 32]);  view_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_408: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None
    
    # No stacktrace found for following nodes
    clone_default_11: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_9, clone_default_10, clone_default_11, None, False, scale = 0.17677669529663687);  clone_default_9 = clone_default_10 = clone_default_11 = None
    getitem_5: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_410: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_821: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_410, [1, 128, 128]);  permute_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_822: "f32[128, 128]" = torch.ops.aten.reshape.default(view_821, [128, 128]);  view_821 = None
    permute_411: "f32[128, 128]" = torch.ops.aten.permute.default(arg1004_1, [1, 0]);  arg1004_1 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[128, 128]" = torch.ops.aten.mm.default(view_822, permute_411);  view_822 = permute_411 = None
    add_tensor_56: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_56, arg1005_1);  mm_default_56 = arg1005_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_823: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_56, [1, 128, 128]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_802: "f32[128, 512]" = torch.ops.aten.reshape.default(add_302, [128, 512])
    permute_401: "f32[512, 128]" = torch.ops.aten.permute.default(arg994_1, [1, 0]);  arg994_1 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[128, 128]" = torch.ops.aten.mm.default(view_802, permute_401);  view_802 = permute_401 = None
    add_tensor_55: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_55, arg995_1);  mm_default_55 = arg995_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_803: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_55, [1, 128, 128]);  add_tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_162: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_803, arg322_1);  view_803 = arg322_1 = None
    add_303: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_162, arg323_1);  mul_162 = arg323_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_306: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_823, add_303);  view_823 = add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_164: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_306, arg326_1);  add_306 = arg326_1 = None
    add_307: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_164, arg327_1);  mul_164 = arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_824: "f32[128, 128]" = torch.ops.aten.reshape.default(add_307, [128, 128])
    permute_412: "f32[128, 512]" = torch.ops.aten.permute.default(arg1006_1, [1, 0]);  arg1006_1 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[128, 512]" = torch.ops.aten.mm.default(view_824, permute_412);  view_824 = permute_412 = None
    add_tensor_54: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_54, arg1007_1);  mm_default_54 = arg1007_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_825: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_54, [1, 128, 512]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_80: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_825);  view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_826: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_80, [128, 512]);  relu_80 = None
    permute_413: "f32[512, 128]" = torch.ops.aten.permute.default(arg1008_1, [1, 0]);  arg1008_1 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[128, 128]" = torch.ops.aten.mm.default(view_826, permute_413);  view_826 = permute_413 = None
    add_tensor_53: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_53, arg1009_1);  mm_default_53 = arg1009_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_827: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_53, [1, 128, 128]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_308: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_827, add_307);  view_827 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_165: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_308, arg328_1);  add_308 = arg328_1 = None
    add_309: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_165, arg329_1);  mul_165 = arg329_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_828: "f32[128, 128]" = torch.ops.aten.reshape.default(add_309, [128, 128])
    permute_414: "f32[128, 512]" = torch.ops.aten.permute.default(arg1010_1, [1, 0]);  arg1010_1 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[128, 512]" = torch.ops.aten.mm.default(view_828, permute_414);  view_828 = permute_414 = None
    add_tensor_52: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_52, arg1011_1);  mm_default_52 = arg1011_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_829: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_52, [1, 128, 512]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_81: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_829);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_830: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_81, [128, 512]);  relu_81 = None
    permute_415: "f32[512, 128]" = torch.ops.aten.permute.default(arg1012_1, [1, 0]);  arg1012_1 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[128, 128]" = torch.ops.aten.mm.default(view_830, permute_415);  view_830 = permute_415 = None
    add_tensor_51: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_51, arg1013_1);  mm_default_51 = arg1013_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_831: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_51, [1, 128, 128]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_310: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_831, add_309);  view_831 = add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_166: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_310, arg330_1);  add_310 = arg330_1 = None
    add_311: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_166, arg331_1);  mul_166 = arg331_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_832: "f32[128, 128]" = torch.ops.aten.reshape.default(add_311, [128, 128])
    permute_416: "f32[128, 512]" = torch.ops.aten.permute.default(arg1014_1, [1, 0]);  arg1014_1 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[128, 512]" = torch.ops.aten.mm.default(view_832, permute_416);  view_832 = permute_416 = None
    add_tensor_50: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_50, arg1015_1);  mm_default_50 = arg1015_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_833: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_50, [1, 128, 512]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_82: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_833);  view_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_834: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_82, [128, 512]);  relu_82 = None
    permute_417: "f32[512, 128]" = torch.ops.aten.permute.default(arg1016_1, [1, 0]);  arg1016_1 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[128, 128]" = torch.ops.aten.mm.default(view_834, permute_417);  view_834 = permute_417 = None
    add_tensor_49: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_49, arg1017_1);  mm_default_49 = arg1017_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_835: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_49, [1, 128, 128]);  add_tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_312: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_835, add_311);  view_835 = add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_167: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_312, arg332_1);  add_312 = arg332_1 = None
    add_313: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_167, arg333_1);  mul_167 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_836: "f32[128, 128]" = torch.ops.aten.reshape.default(add_313, [128, 128])
    permute_418: "f32[128, 512]" = torch.ops.aten.permute.default(arg1018_1, [1, 0]);  arg1018_1 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[128, 512]" = torch.ops.aten.mm.default(view_836, permute_418);  view_836 = permute_418 = None
    add_tensor_48: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_48, arg1019_1);  mm_default_48 = arg1019_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_837: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_48, [1, 128, 512]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_83: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_837);  view_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_838: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_83, [128, 512]);  relu_83 = None
    permute_419: "f32[512, 128]" = torch.ops.aten.permute.default(arg1020_1, [1, 0]);  arg1020_1 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[128, 128]" = torch.ops.aten.mm.default(view_838, permute_419);  view_838 = permute_419 = None
    add_tensor_47: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_47, arg1021_1);  mm_default_47 = arg1021_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_839: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_47, [1, 128, 128]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_314: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_839, add_313);  view_839 = add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_168: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_314, arg334_1);  add_314 = arg334_1 = None
    add_315: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_168, arg335_1);  mul_168 = arg335_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_840: "f32[128, 128]" = torch.ops.aten.reshape.default(add_315, [128, 128]);  add_315 = None
    permute_420: "f32[128, 512]" = torch.ops.aten.permute.default(arg1022_1, [1, 0]);  arg1022_1 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[128, 512]" = torch.ops.aten.mm.default(view_840, permute_420);  view_840 = permute_420 = None
    add_tensor_46: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_46, arg1023_1);  mm_default_46 = arg1023_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_841: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_46, [1, 128, 512]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_316: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_841, add_302);  view_841 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_169: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_316, arg336_1);  add_316 = arg336_1 = None
    add_317: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_169, arg337_1);  mul_169 = arg337_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_844: "f32[128, 512]" = torch.ops.aten.reshape.default(add_317, [128, 512])
    permute_422: "f32[512, 128]" = torch.ops.aten.permute.default(arg1026_1, [1, 0]);  arg1026_1 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[128, 128]" = torch.ops.aten.mm.default(view_844, permute_422);  view_844 = permute_422 = None
    add_tensor_45: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_45, arg1027_1);  mm_default_45 = arg1027_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_845: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_45, [1, 128, 128]);  add_tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_171: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_845, arg340_1);  view_845 = arg340_1 = None
    add_319: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_171, arg341_1);  mul_171 = arg341_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_846: "f32[128, 128]" = torch.ops.aten.reshape.default(add_319, [128, 128])
    permute_423: "f32[128, 128]" = torch.ops.aten.permute.default(arg1028_1, [1, 0]);  arg1028_1 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[128, 128]" = torch.ops.aten.mm.default(view_846, permute_423);  view_846 = permute_423 = None
    add_tensor_44: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_44, arg1029_1);  mm_default_44 = arg1029_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_847: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_44, [1, 128, 128]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_852: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_847, [1, 128, 4, 32]);  view_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
    
    # No stacktrace found for following nodes
    clone_default_6: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_848: "f32[128, 128]" = torch.ops.aten.reshape.default(add_319, [128, 128]);  add_319 = None
    permute_424: "f32[128, 128]" = torch.ops.aten.permute.default(arg1030_1, [1, 0]);  arg1030_1 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[128, 128]" = torch.ops.aten.mm.default(view_848, permute_424);  view_848 = permute_424 = None
    add_tensor_43: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_43, arg1031_1);  mm_default_43 = arg1031_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_849: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_43, [1, 128, 128]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_853: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_849, [1, 128, 4, 32]);  view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_427: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_853, [0, 2, 1, 3]);  view_853 = None
    
    # No stacktrace found for following nodes
    clone_default_7: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_850: "f32[128, 512]" = torch.ops.aten.reshape.default(add_317, [128, 512])
    permute_425: "f32[512, 128]" = torch.ops.aten.permute.default(arg1032_1, [1, 0]);  arg1032_1 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[128, 128]" = torch.ops.aten.mm.default(view_850, permute_425);  view_850 = permute_425 = None
    add_tensor_42: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_42, arg1033_1);  mm_default_42 = arg1033_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_851: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_42, [1, 128, 128]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_854: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_851, [1, 128, 4, 32]);  view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_428: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None
    
    # No stacktrace found for following nodes
    clone_default_8: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_6, clone_default_7, clone_default_8, None, False, scale = 0.17677669529663687);  clone_default_6 = clone_default_7 = clone_default_8 = None
    getitem_4: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_430: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_861: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_430, [1, 128, 128]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_862: "f32[128, 128]" = torch.ops.aten.reshape.default(view_861, [128, 128]);  view_861 = None
    permute_431: "f32[128, 128]" = torch.ops.aten.permute.default(arg1034_1, [1, 0]);  arg1034_1 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[128, 128]" = torch.ops.aten.mm.default(view_862, permute_431);  view_862 = permute_431 = None
    add_tensor_41: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_41, arg1035_1);  mm_default_41 = arg1035_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_863: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_41, [1, 128, 128]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_842: "f32[128, 512]" = torch.ops.aten.reshape.default(add_317, [128, 512])
    permute_421: "f32[512, 128]" = torch.ops.aten.permute.default(arg1024_1, [1, 0]);  arg1024_1 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[128, 128]" = torch.ops.aten.mm.default(view_842, permute_421);  view_842 = permute_421 = None
    add_tensor_40: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_40, arg1025_1);  mm_default_40 = arg1025_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_843: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_40, [1, 128, 128]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_170: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_843, arg338_1);  view_843 = arg338_1 = None
    add_318: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_170, arg339_1);  mul_170 = arg339_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_321: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_863, add_318);  view_863 = add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_172: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_321, arg342_1);  add_321 = arg342_1 = None
    add_322: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_172, arg343_1);  mul_172 = arg343_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_864: "f32[128, 128]" = torch.ops.aten.reshape.default(add_322, [128, 128])
    permute_432: "f32[128, 512]" = torch.ops.aten.permute.default(arg1036_1, [1, 0]);  arg1036_1 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[128, 512]" = torch.ops.aten.mm.default(view_864, permute_432);  view_864 = permute_432 = None
    add_tensor_39: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_39, arg1037_1);  mm_default_39 = arg1037_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_865: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_39, [1, 128, 512]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_84: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_865);  view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_866: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_84, [128, 512]);  relu_84 = None
    permute_433: "f32[512, 128]" = torch.ops.aten.permute.default(arg1038_1, [1, 0]);  arg1038_1 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[128, 128]" = torch.ops.aten.mm.default(view_866, permute_433);  view_866 = permute_433 = None
    add_tensor_38: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_38, arg1039_1);  mm_default_38 = arg1039_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_867: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_38, [1, 128, 128]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_323: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_867, add_322);  view_867 = add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_173: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_323, arg344_1);  add_323 = arg344_1 = None
    add_324: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_173, arg345_1);  mul_173 = arg345_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_868: "f32[128, 128]" = torch.ops.aten.reshape.default(add_324, [128, 128])
    permute_434: "f32[128, 512]" = torch.ops.aten.permute.default(arg1040_1, [1, 0]);  arg1040_1 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[128, 512]" = torch.ops.aten.mm.default(view_868, permute_434);  view_868 = permute_434 = None
    add_tensor_37: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_37, arg1041_1);  mm_default_37 = arg1041_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_869: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_37, [1, 128, 512]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_85: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_869);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_870: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_85, [128, 512]);  relu_85 = None
    permute_435: "f32[512, 128]" = torch.ops.aten.permute.default(arg1042_1, [1, 0]);  arg1042_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[128, 128]" = torch.ops.aten.mm.default(view_870, permute_435);  view_870 = permute_435 = None
    add_tensor_36: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_36, arg1043_1);  mm_default_36 = arg1043_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_871: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_36, [1, 128, 128]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_325: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_871, add_324);  view_871 = add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_174: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_325, arg346_1);  add_325 = arg346_1 = None
    add_326: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_174, arg347_1);  mul_174 = arg347_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_872: "f32[128, 128]" = torch.ops.aten.reshape.default(add_326, [128, 128])
    permute_436: "f32[128, 512]" = torch.ops.aten.permute.default(arg1044_1, [1, 0]);  arg1044_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[128, 512]" = torch.ops.aten.mm.default(view_872, permute_436);  view_872 = permute_436 = None
    add_tensor_35: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_35, arg1045_1);  mm_default_35 = arg1045_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_873: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 128, 512]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_86: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_873);  view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_874: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_86, [128, 512]);  relu_86 = None
    permute_437: "f32[512, 128]" = torch.ops.aten.permute.default(arg1046_1, [1, 0]);  arg1046_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[128, 128]" = torch.ops.aten.mm.default(view_874, permute_437);  view_874 = permute_437 = None
    add_tensor_34: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_34, arg1047_1);  mm_default_34 = arg1047_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_875: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 128, 128]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_327: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_875, add_326);  view_875 = add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_175: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_327, arg348_1);  add_327 = arg348_1 = None
    add_328: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_175, arg349_1);  mul_175 = arg349_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_876: "f32[128, 128]" = torch.ops.aten.reshape.default(add_328, [128, 128])
    permute_438: "f32[128, 512]" = torch.ops.aten.permute.default(arg1048_1, [1, 0]);  arg1048_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[128, 512]" = torch.ops.aten.mm.default(view_876, permute_438);  view_876 = permute_438 = None
    add_tensor_33: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_33, arg1049_1);  mm_default_33 = arg1049_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_877: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 128, 512]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_87: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_877);  view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_878: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_87, [128, 512]);  relu_87 = None
    permute_439: "f32[512, 128]" = torch.ops.aten.permute.default(arg1050_1, [1, 0]);  arg1050_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[128, 128]" = torch.ops.aten.mm.default(view_878, permute_439);  view_878 = permute_439 = None
    add_tensor_32: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_32, arg1051_1);  mm_default_32 = arg1051_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_879: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 128, 128]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_329: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_879, add_328);  view_879 = add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_176: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_329, arg350_1);  add_329 = arg350_1 = None
    add_330: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_176, arg351_1);  mul_176 = arg351_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_880: "f32[128, 128]" = torch.ops.aten.reshape.default(add_330, [128, 128]);  add_330 = None
    permute_440: "f32[128, 512]" = torch.ops.aten.permute.default(arg1052_1, [1, 0]);  arg1052_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[128, 512]" = torch.ops.aten.mm.default(view_880, permute_440);  view_880 = permute_440 = None
    add_tensor_31: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_31, arg1053_1);  mm_default_31 = arg1053_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_881: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 128, 512]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_331: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_881, add_317);  view_881 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_177: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_331, arg352_1);  add_331 = arg352_1 = None
    add_332: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_177, arg353_1);  mul_177 = arg353_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_884: "f32[128, 512]" = torch.ops.aten.reshape.default(add_332, [128, 512])
    permute_442: "f32[512, 128]" = torch.ops.aten.permute.default(arg1056_1, [1, 0]);  arg1056_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[128, 128]" = torch.ops.aten.mm.default(view_884, permute_442);  view_884 = permute_442 = None
    add_tensor_30: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_30, arg1057_1);  mm_default_30 = arg1057_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_885: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 128, 128]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_179: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_885, arg356_1);  view_885 = arg356_1 = None
    add_334: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_179, arg357_1);  mul_179 = arg357_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_886: "f32[128, 128]" = torch.ops.aten.reshape.default(add_334, [128, 128])
    permute_443: "f32[128, 128]" = torch.ops.aten.permute.default(arg1058_1, [1, 0]);  arg1058_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[128, 128]" = torch.ops.aten.mm.default(view_886, permute_443);  view_886 = permute_443 = None
    add_tensor_29: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_29, arg1059_1);  mm_default_29 = arg1059_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_887: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 128, 128]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_892: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_887, [1, 128, 4, 32]);  view_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_446: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
    
    # No stacktrace found for following nodes
    clone_default_3: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_888: "f32[128, 128]" = torch.ops.aten.reshape.default(add_334, [128, 128]);  add_334 = None
    permute_444: "f32[128, 128]" = torch.ops.aten.permute.default(arg1060_1, [1, 0]);  arg1060_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[128, 128]" = torch.ops.aten.mm.default(view_888, permute_444);  view_888 = permute_444 = None
    add_tensor_28: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_28, arg1061_1);  mm_default_28 = arg1061_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_889: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 128, 128]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_893: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_889, [1, 128, 4, 32]);  view_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_447: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None
    
    # No stacktrace found for following nodes
    clone_default_4: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_890: "f32[128, 512]" = torch.ops.aten.reshape.default(add_332, [128, 512])
    permute_445: "f32[512, 128]" = torch.ops.aten.permute.default(arg1062_1, [1, 0]);  arg1062_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[128, 128]" = torch.ops.aten.mm.default(view_890, permute_445);  view_890 = permute_445 = None
    add_tensor_27: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_27, arg1063_1);  mm_default_27 = arg1063_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_891: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 128, 128]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_894: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_891, [1, 128, 4, 32]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_448: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_894, [0, 2, 1, 3]);  view_894 = None
    
    # No stacktrace found for following nodes
    clone_default_5: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_3, clone_default_4, clone_default_5, None, False, scale = 0.17677669529663687);  clone_default_3 = clone_default_4 = clone_default_5 = None
    getitem_3: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_450: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_3, [0, 2, 1, 3]);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_901: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_450, [1, 128, 128]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_902: "f32[128, 128]" = torch.ops.aten.reshape.default(view_901, [128, 128]);  view_901 = None
    permute_451: "f32[128, 128]" = torch.ops.aten.permute.default(arg1064_1, [1, 0]);  arg1064_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[128, 128]" = torch.ops.aten.mm.default(view_902, permute_451);  view_902 = permute_451 = None
    add_tensor_26: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_26, arg1065_1);  mm_default_26 = arg1065_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_903: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 128, 128]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_882: "f32[128, 512]" = torch.ops.aten.reshape.default(add_332, [128, 512])
    permute_441: "f32[512, 128]" = torch.ops.aten.permute.default(arg1054_1, [1, 0]);  arg1054_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[128, 128]" = torch.ops.aten.mm.default(view_882, permute_441);  view_882 = permute_441 = None
    add_tensor_25: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_25, arg1055_1);  mm_default_25 = arg1055_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_883: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 128, 128]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_178: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_883, arg354_1);  view_883 = arg354_1 = None
    add_333: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_178, arg355_1);  mul_178 = arg355_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_336: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_903, add_333);  view_903 = add_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_180: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_336, arg358_1);  add_336 = arg358_1 = None
    add_337: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_180, arg359_1);  mul_180 = arg359_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_904: "f32[128, 128]" = torch.ops.aten.reshape.default(add_337, [128, 128])
    permute_452: "f32[128, 512]" = torch.ops.aten.permute.default(arg1066_1, [1, 0]);  arg1066_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[128, 512]" = torch.ops.aten.mm.default(view_904, permute_452);  view_904 = permute_452 = None
    add_tensor_24: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_24, arg1067_1);  mm_default_24 = arg1067_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_905: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 128, 512]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_88: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_905);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_906: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_88, [128, 512]);  relu_88 = None
    permute_453: "f32[512, 128]" = torch.ops.aten.permute.default(arg1068_1, [1, 0]);  arg1068_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[128, 128]" = torch.ops.aten.mm.default(view_906, permute_453);  view_906 = permute_453 = None
    add_tensor_23: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_23, arg1069_1);  mm_default_23 = arg1069_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_907: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 128, 128]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_338: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_907, add_337);  view_907 = add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_181: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_338, arg360_1);  add_338 = arg360_1 = None
    add_339: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_181, arg361_1);  mul_181 = arg361_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_908: "f32[128, 128]" = torch.ops.aten.reshape.default(add_339, [128, 128])
    permute_454: "f32[128, 512]" = torch.ops.aten.permute.default(arg1070_1, [1, 0]);  arg1070_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[128, 512]" = torch.ops.aten.mm.default(view_908, permute_454);  view_908 = permute_454 = None
    add_tensor_22: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_22, arg1071_1);  mm_default_22 = arg1071_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_909: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 128, 512]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_89: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_909);  view_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_910: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_89, [128, 512]);  relu_89 = None
    permute_455: "f32[512, 128]" = torch.ops.aten.permute.default(arg1072_1, [1, 0]);  arg1072_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[128, 128]" = torch.ops.aten.mm.default(view_910, permute_455);  view_910 = permute_455 = None
    add_tensor_21: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_21, arg1073_1);  mm_default_21 = arg1073_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_911: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 128, 128]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_340: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_911, add_339);  view_911 = add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_182: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_340, arg362_1);  add_340 = arg362_1 = None
    add_341: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_182, arg363_1);  mul_182 = arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_912: "f32[128, 128]" = torch.ops.aten.reshape.default(add_341, [128, 128])
    permute_456: "f32[128, 512]" = torch.ops.aten.permute.default(arg1074_1, [1, 0]);  arg1074_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[128, 512]" = torch.ops.aten.mm.default(view_912, permute_456);  view_912 = permute_456 = None
    add_tensor_20: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_20, arg1075_1);  mm_default_20 = arg1075_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_913: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 128, 512]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_90: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_913);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_914: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_90, [128, 512]);  relu_90 = None
    permute_457: "f32[512, 128]" = torch.ops.aten.permute.default(arg1076_1, [1, 0]);  arg1076_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[128, 128]" = torch.ops.aten.mm.default(view_914, permute_457);  view_914 = permute_457 = None
    add_tensor_19: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_19, arg1077_1);  mm_default_19 = arg1077_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_915: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 128, 128]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_342: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_915, add_341);  view_915 = add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_183: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_342, arg364_1);  add_342 = arg364_1 = None
    add_343: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_183, arg365_1);  mul_183 = arg365_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_916: "f32[128, 128]" = torch.ops.aten.reshape.default(add_343, [128, 128])
    permute_458: "f32[128, 512]" = torch.ops.aten.permute.default(arg1078_1, [1, 0]);  arg1078_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[128, 512]" = torch.ops.aten.mm.default(view_916, permute_458);  view_916 = permute_458 = None
    add_tensor_18: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_18, arg1079_1);  mm_default_18 = arg1079_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_917: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 128, 512]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_91: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_917);  view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_918: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_91, [128, 512]);  relu_91 = None
    permute_459: "f32[512, 128]" = torch.ops.aten.permute.default(arg1080_1, [1, 0]);  arg1080_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[128, 128]" = torch.ops.aten.mm.default(view_918, permute_459);  view_918 = permute_459 = None
    add_tensor_17: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_17, arg1081_1);  mm_default_17 = arg1081_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_919: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 128, 128]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_344: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_919, add_343);  view_919 = add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_184: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_344, arg366_1);  add_344 = arg366_1 = None
    add_345: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_184, arg367_1);  mul_184 = arg367_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_920: "f32[128, 128]" = torch.ops.aten.reshape.default(add_345, [128, 128]);  add_345 = None
    permute_460: "f32[128, 512]" = torch.ops.aten.permute.default(arg1082_1, [1, 0]);  arg1082_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[128, 512]" = torch.ops.aten.mm.default(view_920, permute_460);  view_920 = permute_460 = None
    add_tensor_16: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_16, arg1083_1);  mm_default_16 = arg1083_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_921: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 128, 512]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_346: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_921, add_332);  view_921 = add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_185: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_346, arg368_1);  add_346 = arg368_1 = None
    add_347: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_185, arg369_1);  mul_185 = arg369_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_924: "f32[128, 512]" = torch.ops.aten.reshape.default(add_347, [128, 512])
    permute_462: "f32[512, 128]" = torch.ops.aten.permute.default(arg1086_1, [1, 0]);  arg1086_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[128, 128]" = torch.ops.aten.mm.default(view_924, permute_462);  view_924 = permute_462 = None
    add_tensor_15: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_15, arg1087_1);  mm_default_15 = arg1087_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_925: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 128, 128]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_187: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_925, arg372_1);  view_925 = arg372_1 = None
    add_349: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_187, arg373_1);  mul_187 = arg373_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_926: "f32[128, 128]" = torch.ops.aten.reshape.default(add_349, [128, 128])
    permute_463: "f32[128, 128]" = torch.ops.aten.permute.default(arg1088_1, [1, 0]);  arg1088_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[128, 128]" = torch.ops.aten.mm.default(view_926, permute_463);  view_926 = permute_463 = None
    add_tensor_14: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_14, arg1089_1);  mm_default_14 = arg1089_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_927: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 128, 128]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_932: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_927, [1, 128, 4, 32]);  view_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_466: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_932, [0, 2, 1, 3]);  view_932 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_928: "f32[128, 128]" = torch.ops.aten.reshape.default(add_349, [128, 128]);  add_349 = None
    permute_464: "f32[128, 128]" = torch.ops.aten.permute.default(arg1090_1, [1, 0]);  arg1090_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[128, 128]" = torch.ops.aten.mm.default(view_928, permute_464);  view_928 = permute_464 = None
    add_tensor_13: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_13, arg1091_1);  mm_default_13 = arg1091_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_929: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 128, 128]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_933: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_929, [1, 128, 4, 32]);  view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_467: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_933, [0, 2, 1, 3]);  view_933 = None
    
    # No stacktrace found for following nodes
    clone_default_1: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_930: "f32[128, 512]" = torch.ops.aten.reshape.default(add_347, [128, 512])
    permute_465: "f32[512, 128]" = torch.ops.aten.permute.default(arg1092_1, [1, 0]);  arg1092_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[128, 128]" = torch.ops.aten.mm.default(view_930, permute_465);  view_930 = permute_465 = None
    add_tensor_12: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_12, arg1093_1);  mm_default_12 = arg1093_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_931: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 128, 128]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_934: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_931, [1, 128, 4, 32]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_468: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None
    
    # No stacktrace found for following nodes
    clone_default_2: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default, clone_default_1, clone_default_2, None, False, scale = 0.17677669529663687);  clone_default = clone_default_1 = clone_default_2 = None
    getitem_2: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_470: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_941: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_470, [1, 128, 128]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_942: "f32[128, 128]" = torch.ops.aten.reshape.default(view_941, [128, 128]);  view_941 = None
    permute_471: "f32[128, 128]" = torch.ops.aten.permute.default(arg1094_1, [1, 0]);  arg1094_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[128, 128]" = torch.ops.aten.mm.default(view_942, permute_471);  view_942 = permute_471 = None
    add_tensor_11: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_11, arg1095_1);  mm_default_11 = arg1095_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_943: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 128, 128]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_922: "f32[128, 512]" = torch.ops.aten.reshape.default(add_347, [128, 512])
    permute_461: "f32[512, 128]" = torch.ops.aten.permute.default(arg1084_1, [1, 0]);  arg1084_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[128, 128]" = torch.ops.aten.mm.default(view_922, permute_461);  view_922 = permute_461 = None
    add_tensor_10: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_10, arg1085_1);  mm_default_10 = arg1085_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_923: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 128, 128]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_186: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_923, arg370_1);  view_923 = arg370_1 = None
    add_348: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_186, arg371_1);  mul_186 = arg371_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_351: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_943, add_348);  view_943 = add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_188: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_351, arg374_1);  add_351 = arg374_1 = None
    add_352: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_188, arg375_1);  mul_188 = arg375_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_944: "f32[128, 128]" = torch.ops.aten.reshape.default(add_352, [128, 128])
    permute_472: "f32[128, 512]" = torch.ops.aten.permute.default(arg1096_1, [1, 0]);  arg1096_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[128, 512]" = torch.ops.aten.mm.default(view_944, permute_472);  view_944 = permute_472 = None
    add_tensor_9: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_9, arg1097_1);  mm_default_9 = arg1097_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_945: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 128, 512]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_92: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_945);  view_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_946: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_92, [128, 512]);  relu_92 = None
    permute_473: "f32[512, 128]" = torch.ops.aten.permute.default(arg1098_1, [1, 0]);  arg1098_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[128, 128]" = torch.ops.aten.mm.default(view_946, permute_473);  view_946 = permute_473 = None
    add_tensor_8: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_8, arg1099_1);  mm_default_8 = arg1099_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_947: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 128, 128]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_353: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_947, add_352);  view_947 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_189: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_353, arg376_1);  add_353 = arg376_1 = None
    add_354: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, arg377_1);  mul_189 = arg377_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_948: "f32[128, 128]" = torch.ops.aten.reshape.default(add_354, [128, 128])
    permute_474: "f32[128, 512]" = torch.ops.aten.permute.default(arg1100_1, [1, 0]);  arg1100_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[128, 512]" = torch.ops.aten.mm.default(view_948, permute_474);  view_948 = permute_474 = None
    add_tensor_7: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_7, arg1101_1);  mm_default_7 = arg1101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_949: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 128, 512]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_93: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_949);  view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_950: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_93, [128, 512]);  relu_93 = None
    permute_475: "f32[512, 128]" = torch.ops.aten.permute.default(arg1102_1, [1, 0]);  arg1102_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[128, 128]" = torch.ops.aten.mm.default(view_950, permute_475);  view_950 = permute_475 = None
    add_tensor_6: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_6, arg1103_1);  mm_default_6 = arg1103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_951: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 128, 128]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_355: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_951, add_354);  view_951 = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_190: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_355, arg378_1);  add_355 = arg378_1 = None
    add_356: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_190, arg379_1);  mul_190 = arg379_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_952: "f32[128, 128]" = torch.ops.aten.reshape.default(add_356, [128, 128])
    permute_476: "f32[128, 512]" = torch.ops.aten.permute.default(arg1104_1, [1, 0]);  arg1104_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[128, 512]" = torch.ops.aten.mm.default(view_952, permute_476);  view_952 = permute_476 = None
    add_tensor_5: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_5, arg1105_1);  mm_default_5 = arg1105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_953: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 128, 512]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_94: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_953);  view_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_954: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_94, [128, 512]);  relu_94 = None
    permute_477: "f32[512, 128]" = torch.ops.aten.permute.default(arg1106_1, [1, 0]);  arg1106_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[128, 128]" = torch.ops.aten.mm.default(view_954, permute_477);  view_954 = permute_477 = None
    add_tensor_4: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_4, arg1107_1);  mm_default_4 = arg1107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_955: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 128, 128]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_357: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_955, add_356);  view_955 = add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_191: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_357, arg380_1);  add_357 = arg380_1 = None
    add_358: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_191, arg381_1);  mul_191 = arg381_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_956: "f32[128, 128]" = torch.ops.aten.reshape.default(add_358, [128, 128])
    permute_478: "f32[128, 512]" = torch.ops.aten.permute.default(arg1108_1, [1, 0]);  arg1108_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[128, 512]" = torch.ops.aten.mm.default(view_956, permute_478);  view_956 = permute_478 = None
    add_tensor_3: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_3, arg1109_1);  mm_default_3 = arg1109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_957: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 128, 512]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_95: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_957);  view_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_958: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_95, [128, 512]);  relu_95 = None
    permute_479: "f32[512, 128]" = torch.ops.aten.permute.default(arg1110_1, [1, 0]);  arg1110_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[128, 128]" = torch.ops.aten.mm.default(view_958, permute_479);  view_958 = permute_479 = None
    add_tensor_2: "f32[128, 128]" = torch.ops.aten.add.Tensor(mm_default_2, arg1111_1);  mm_default_2 = arg1111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_959: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 128, 128]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_359: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_959, add_358);  view_959 = add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_192: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_359, arg382_1);  add_359 = arg382_1 = None
    add_360: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_192, arg383_1);  mul_192 = arg383_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_960: "f32[128, 128]" = torch.ops.aten.reshape.default(add_360, [128, 128]);  add_360 = None
    permute_480: "f32[128, 512]" = torch.ops.aten.permute.default(arg1112_1, [1, 0]);  arg1112_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[128, 512]" = torch.ops.aten.mm.default(view_960, permute_480);  view_960 = permute_480 = None
    add_tensor_1: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_1, arg1113_1);  mm_default_1 = arg1113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_961: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 128, 512]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_361: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_961, add_347);  view_961 = add_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_193: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_361, arg384_1);  add_361 = arg384_1 = None
    add_362: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_193, arg385_1);  mul_193 = arg385_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:634, code: hidden_states = self.dense(hidden_states)
    view_962: "f32[128, 512]" = torch.ops.aten.reshape.default(add_362, [128, 512]);  add_362 = None
    permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(arg1114_1, [1, 0]);  arg1114_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[128, 512]" = torch.ops.aten.mm.default(view_962, permute_481);  view_962 = permute_481 = None
    add_tensor: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default, arg1115_1);  mm_default = arg1115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:634, code: hidden_states = self.dense(hidden_states)
    view_963: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor, [1, 128, 512]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:635, code: hidden_states = self.transform_act_fn(hidden_states)
    relu_96: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_963);  view_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:636, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(relu_96, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1107, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_969: "i64[128]" = torch.ops.aten.reshape.default(arg1120_1, [-1]);  arg1120_1 = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_969, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:636, code: hidden_states = self.LayerNorm(hidden_states)
    sub_25: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(relu_96, getitem_1);  relu_96 = getitem_1 = None
    add_363: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_363);  add_363 = None
    mul_194: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt);  sub_25 = rsqrt = None
    mul_195: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_194, arg1116_1);  mul_194 = arg1116_1 = None
    add_364: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_195, arg1117_1);  mul_195 = arg1117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:654, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
    view_964: "f32[128, 512]" = torch.ops.aten.reshape.default(add_364, [128, 512]);  add_364 = None
    permute_482: "f32[128, 30522]" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
    cat_1: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_482, arg387_1]);  permute_482 = arg387_1 = None
    mm: "f32[128, 30522]" = torch.ops.aten.mm.default(view_964, cat_1);  view_964 = cat_1 = None
    view_965: "f32[1, 128, 30522]" = torch.ops.aten.reshape.default(mm, [1, 128, 30522]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:655, code: hidden_states += self.decoder.bias
    add_365: "f32[1, 128, 30522]" = torch.ops.aten.add.Tensor(view_965, arg388_1);  view_965 = arg388_1 = None
    view_966: "f32[128, 30522]" = torch.ops.aten.reshape.default(add_365, [128, 30522]);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1107, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_970: "f32[1, 128, 30522]" = torch.ops.aten.reshape.default(view_966, [1, 128, 30522])
    view_971: "f32[128, 30522]" = torch.ops.aten.reshape.default(view_970, [-1, 30522]);  view_970 = None
    amax_24: "f32[128, 1]" = torch.ops.aten.amax.default(view_971, [1], True)
    sub_26: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(view_971, amax_24);  view_971 = amax_24 = None
    exp_24: "f32[128, 30522]" = torch.ops.aten.exp.default(sub_26)
    sum_25: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_27: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_969, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[128]" = torch.ops.aten.where.self(ne, view_969, full_default_2);  ne = full_default_2 = None
    unsqueeze_2: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_2);  sub_27 = unsqueeze_2 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_969, -100);  view_969 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:655, code: hidden_states += self.decoder.bias
    view_967: "f32[1, 128, 30522]" = torch.ops.aten.reshape.default(view_966, [1, 128, 30522]);  view_966 = None
    return (div_48, view_967)
    