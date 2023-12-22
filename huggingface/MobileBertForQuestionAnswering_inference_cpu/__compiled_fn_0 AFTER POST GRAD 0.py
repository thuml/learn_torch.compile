from __future__ import annotations



def forward(self, arg0_1: "f32[512]", arg1_1: "f32[512]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[128]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[512]", arg17_1: "f32[512]", arg18_1: "f32[128]", arg19_1: "f32[128]", arg20_1: "f32[128]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[128]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[128]", arg35_1: "f32[128]", arg36_1: "f32[128]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[128]", arg47_1: "f32[128]", arg48_1: "f32[512]", arg49_1: "f32[512]", arg50_1: "f32[128]", arg51_1: "f32[128]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[128]", arg55_1: "f32[128]", arg56_1: "f32[128]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[128]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[128]", arg67_1: "f32[128]", arg68_1: "f32[128]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[128]", arg72_1: "f32[128]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[128]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128]", arg87_1: "f32[128]", arg88_1: "f32[128]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[128]", arg92_1: "f32[128]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[512]", arg97_1: "f32[512]", arg98_1: "f32[128]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128]", arg106_1: "f32[128]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[128]", arg111_1: "f32[128]", arg112_1: "f32[512]", arg113_1: "f32[512]", arg114_1: "f32[128]", arg115_1: "f32[128]", arg116_1: "f32[128]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[128]", arg122_1: "f32[128]", arg123_1: "f32[128]", arg124_1: "f32[128]", arg125_1: "f32[128]", arg126_1: "f32[128]", arg127_1: "f32[128]", arg128_1: "f32[512]", arg129_1: "f32[512]", arg130_1: "f32[128]", arg131_1: "f32[128]", arg132_1: "f32[128]", arg133_1: "f32[128]", arg134_1: "f32[128]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[128]", arg138_1: "f32[128]", arg139_1: "f32[128]", arg140_1: "f32[128]", arg141_1: "f32[128]", arg142_1: "f32[128]", arg143_1: "f32[128]", arg144_1: "f32[512]", arg145_1: "f32[512]", arg146_1: "f32[128]", arg147_1: "f32[128]", arg148_1: "f32[128]", arg149_1: "f32[128]", arg150_1: "f32[128]", arg151_1: "f32[128]", arg152_1: "f32[128]", arg153_1: "f32[128]", arg154_1: "f32[128]", arg155_1: "f32[128]", arg156_1: "f32[128]", arg157_1: "f32[128]", arg158_1: "f32[128]", arg159_1: "f32[128]", arg160_1: "f32[512]", arg161_1: "f32[512]", arg162_1: "f32[128]", arg163_1: "f32[128]", arg164_1: "f32[128]", arg165_1: "f32[128]", arg166_1: "f32[128]", arg167_1: "f32[128]", arg168_1: "f32[128]", arg169_1: "f32[128]", arg170_1: "f32[128]", arg171_1: "f32[128]", arg172_1: "f32[128]", arg173_1: "f32[128]", arg174_1: "f32[128]", arg175_1: "f32[128]", arg176_1: "f32[512]", arg177_1: "f32[512]", arg178_1: "f32[128]", arg179_1: "f32[128]", arg180_1: "f32[128]", arg181_1: "f32[128]", arg182_1: "f32[128]", arg183_1: "f32[128]", arg184_1: "f32[128]", arg185_1: "f32[128]", arg186_1: "f32[128]", arg187_1: "f32[128]", arg188_1: "f32[128]", arg189_1: "f32[128]", arg190_1: "f32[128]", arg191_1: "f32[128]", arg192_1: "f32[512]", arg193_1: "f32[512]", arg194_1: "f32[128]", arg195_1: "f32[128]", arg196_1: "f32[128]", arg197_1: "f32[128]", arg198_1: "f32[128]", arg199_1: "f32[128]", arg200_1: "f32[128]", arg201_1: "f32[128]", arg202_1: "f32[128]", arg203_1: "f32[128]", arg204_1: "f32[128]", arg205_1: "f32[128]", arg206_1: "f32[128]", arg207_1: "f32[128]", arg208_1: "f32[512]", arg209_1: "f32[512]", arg210_1: "f32[128]", arg211_1: "f32[128]", arg212_1: "f32[128]", arg213_1: "f32[128]", arg214_1: "f32[128]", arg215_1: "f32[128]", arg216_1: "f32[128]", arg217_1: "f32[128]", arg218_1: "f32[128]", arg219_1: "f32[128]", arg220_1: "f32[128]", arg221_1: "f32[128]", arg222_1: "f32[128]", arg223_1: "f32[128]", arg224_1: "f32[512]", arg225_1: "f32[512]", arg226_1: "f32[128]", arg227_1: "f32[128]", arg228_1: "f32[128]", arg229_1: "f32[128]", arg230_1: "f32[128]", arg231_1: "f32[128]", arg232_1: "f32[128]", arg233_1: "f32[128]", arg234_1: "f32[128]", arg235_1: "f32[128]", arg236_1: "f32[128]", arg237_1: "f32[128]", arg238_1: "f32[128]", arg239_1: "f32[128]", arg240_1: "f32[512]", arg241_1: "f32[512]", arg242_1: "f32[128]", arg243_1: "f32[128]", arg244_1: "f32[128]", arg245_1: "f32[128]", arg246_1: "f32[128]", arg247_1: "f32[128]", arg248_1: "f32[128]", arg249_1: "f32[128]", arg250_1: "f32[128]", arg251_1: "f32[128]", arg252_1: "f32[128]", arg253_1: "f32[128]", arg254_1: "f32[128]", arg255_1: "f32[128]", arg256_1: "f32[512]", arg257_1: "f32[512]", arg258_1: "f32[128]", arg259_1: "f32[128]", arg260_1: "f32[128]", arg261_1: "f32[128]", arg262_1: "f32[128]", arg263_1: "f32[128]", arg264_1: "f32[128]", arg265_1: "f32[128]", arg266_1: "f32[128]", arg267_1: "f32[128]", arg268_1: "f32[128]", arg269_1: "f32[128]", arg270_1: "f32[128]", arg271_1: "f32[128]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[128]", arg275_1: "f32[128]", arg276_1: "f32[128]", arg277_1: "f32[128]", arg278_1: "f32[128]", arg279_1: "f32[128]", arg280_1: "f32[128]", arg281_1: "f32[128]", arg282_1: "f32[128]", arg283_1: "f32[128]", arg284_1: "f32[128]", arg285_1: "f32[128]", arg286_1: "f32[128]", arg287_1: "f32[128]", arg288_1: "f32[512]", arg289_1: "f32[512]", arg290_1: "f32[128]", arg291_1: "f32[128]", arg292_1: "f32[128]", arg293_1: "f32[128]", arg294_1: "f32[128]", arg295_1: "f32[128]", arg296_1: "f32[128]", arg297_1: "f32[128]", arg298_1: "f32[128]", arg299_1: "f32[128]", arg300_1: "f32[128]", arg301_1: "f32[128]", arg302_1: "f32[128]", arg303_1: "f32[128]", arg304_1: "f32[512]", arg305_1: "f32[512]", arg306_1: "f32[128]", arg307_1: "f32[128]", arg308_1: "f32[128]", arg309_1: "f32[128]", arg310_1: "f32[128]", arg311_1: "f32[128]", arg312_1: "f32[128]", arg313_1: "f32[128]", arg314_1: "f32[128]", arg315_1: "f32[128]", arg316_1: "f32[128]", arg317_1: "f32[128]", arg318_1: "f32[128]", arg319_1: "f32[128]", arg320_1: "f32[512]", arg321_1: "f32[512]", arg322_1: "f32[128]", arg323_1: "f32[128]", arg324_1: "f32[128]", arg325_1: "f32[128]", arg326_1: "f32[128]", arg327_1: "f32[128]", arg328_1: "f32[128]", arg329_1: "f32[128]", arg330_1: "f32[128]", arg331_1: "f32[128]", arg332_1: "f32[128]", arg333_1: "f32[128]", arg334_1: "f32[128]", arg335_1: "f32[128]", arg336_1: "f32[512]", arg337_1: "f32[512]", arg338_1: "f32[128]", arg339_1: "f32[128]", arg340_1: "f32[128]", arg341_1: "f32[128]", arg342_1: "f32[128]", arg343_1: "f32[128]", arg344_1: "f32[128]", arg345_1: "f32[128]", arg346_1: "f32[128]", arg347_1: "f32[128]", arg348_1: "f32[128]", arg349_1: "f32[128]", arg350_1: "f32[128]", arg351_1: "f32[128]", arg352_1: "f32[512]", arg353_1: "f32[512]", arg354_1: "f32[128]", arg355_1: "f32[128]", arg356_1: "f32[128]", arg357_1: "f32[128]", arg358_1: "f32[128]", arg359_1: "f32[128]", arg360_1: "f32[128]", arg361_1: "f32[128]", arg362_1: "f32[128]", arg363_1: "f32[128]", arg364_1: "f32[128]", arg365_1: "f32[128]", arg366_1: "f32[128]", arg367_1: "f32[128]", arg368_1: "f32[512]", arg369_1: "f32[512]", arg370_1: "f32[128]", arg371_1: "f32[128]", arg372_1: "f32[128]", arg373_1: "f32[128]", arg374_1: "f32[128]", arg375_1: "f32[128]", arg376_1: "f32[128]", arg377_1: "f32[128]", arg378_1: "f32[128]", arg379_1: "f32[128]", arg380_1: "f32[128]", arg381_1: "f32[128]", arg382_1: "f32[128]", arg383_1: "f32[128]", arg384_1: "f32[512]", arg385_1: "f32[512]", arg386_1: "f32[30522, 128]", arg387_1: "f32[512, 384]", arg388_1: "f32[512]", arg389_1: "f32[512, 512]", arg390_1: "f32[2, 512]", arg391_1: "f32[128, 512]", arg392_1: "f32[128]", arg393_1: "f32[128, 512]", arg394_1: "f32[128]", arg395_1: "f32[128, 128]", arg396_1: "f32[128]", arg397_1: "f32[128, 128]", arg398_1: "f32[128]", arg399_1: "f32[128, 512]", arg400_1: "f32[128]", arg401_1: "f32[128, 128]", arg402_1: "f32[128]", arg403_1: "f32[512, 128]", arg404_1: "f32[512]", arg405_1: "f32[128, 512]", arg406_1: "f32[128]", arg407_1: "f32[512, 128]", arg408_1: "f32[512]", arg409_1: "f32[128, 512]", arg410_1: "f32[128]", arg411_1: "f32[512, 128]", arg412_1: "f32[512]", arg413_1: "f32[128, 512]", arg414_1: "f32[128]", arg415_1: "f32[512, 128]", arg416_1: "f32[512]", arg417_1: "f32[128, 512]", arg418_1: "f32[128]", arg419_1: "f32[512, 128]", arg420_1: "f32[512]", arg421_1: "f32[128, 512]", arg422_1: "f32[128]", arg423_1: "f32[128, 512]", arg424_1: "f32[128]", arg425_1: "f32[128, 128]", arg426_1: "f32[128]", arg427_1: "f32[128, 128]", arg428_1: "f32[128]", arg429_1: "f32[128, 512]", arg430_1: "f32[128]", arg431_1: "f32[128, 128]", arg432_1: "f32[128]", arg433_1: "f32[512, 128]", arg434_1: "f32[512]", arg435_1: "f32[128, 512]", arg436_1: "f32[128]", arg437_1: "f32[512, 128]", arg438_1: "f32[512]", arg439_1: "f32[128, 512]", arg440_1: "f32[128]", arg441_1: "f32[512, 128]", arg442_1: "f32[512]", arg443_1: "f32[128, 512]", arg444_1: "f32[128]", arg445_1: "f32[512, 128]", arg446_1: "f32[512]", arg447_1: "f32[128, 512]", arg448_1: "f32[128]", arg449_1: "f32[512, 128]", arg450_1: "f32[512]", arg451_1: "f32[128, 512]", arg452_1: "f32[128]", arg453_1: "f32[128, 512]", arg454_1: "f32[128]", arg455_1: "f32[128, 128]", arg456_1: "f32[128]", arg457_1: "f32[128, 128]", arg458_1: "f32[128]", arg459_1: "f32[128, 512]", arg460_1: "f32[128]", arg461_1: "f32[128, 128]", arg462_1: "f32[128]", arg463_1: "f32[512, 128]", arg464_1: "f32[512]", arg465_1: "f32[128, 512]", arg466_1: "f32[128]", arg467_1: "f32[512, 128]", arg468_1: "f32[512]", arg469_1: "f32[128, 512]", arg470_1: "f32[128]", arg471_1: "f32[512, 128]", arg472_1: "f32[512]", arg473_1: "f32[128, 512]", arg474_1: "f32[128]", arg475_1: "f32[512, 128]", arg476_1: "f32[512]", arg477_1: "f32[128, 512]", arg478_1: "f32[128]", arg479_1: "f32[512, 128]", arg480_1: "f32[512]", arg481_1: "f32[128, 512]", arg482_1: "f32[128]", arg483_1: "f32[128, 512]", arg484_1: "f32[128]", arg485_1: "f32[128, 128]", arg486_1: "f32[128]", arg487_1: "f32[128, 128]", arg488_1: "f32[128]", arg489_1: "f32[128, 512]", arg490_1: "f32[128]", arg491_1: "f32[128, 128]", arg492_1: "f32[128]", arg493_1: "f32[512, 128]", arg494_1: "f32[512]", arg495_1: "f32[128, 512]", arg496_1: "f32[128]", arg497_1: "f32[512, 128]", arg498_1: "f32[512]", arg499_1: "f32[128, 512]", arg500_1: "f32[128]", arg501_1: "f32[512, 128]", arg502_1: "f32[512]", arg503_1: "f32[128, 512]", arg504_1: "f32[128]", arg505_1: "f32[512, 128]", arg506_1: "f32[512]", arg507_1: "f32[128, 512]", arg508_1: "f32[128]", arg509_1: "f32[512, 128]", arg510_1: "f32[512]", arg511_1: "f32[128, 512]", arg512_1: "f32[128]", arg513_1: "f32[128, 512]", arg514_1: "f32[128]", arg515_1: "f32[128, 128]", arg516_1: "f32[128]", arg517_1: "f32[128, 128]", arg518_1: "f32[128]", arg519_1: "f32[128, 512]", arg520_1: "f32[128]", arg521_1: "f32[128, 128]", arg522_1: "f32[128]", arg523_1: "f32[512, 128]", arg524_1: "f32[512]", arg525_1: "f32[128, 512]", arg526_1: "f32[128]", arg527_1: "f32[512, 128]", arg528_1: "f32[512]", arg529_1: "f32[128, 512]", arg530_1: "f32[128]", arg531_1: "f32[512, 128]", arg532_1: "f32[512]", arg533_1: "f32[128, 512]", arg534_1: "f32[128]", arg535_1: "f32[512, 128]", arg536_1: "f32[512]", arg537_1: "f32[128, 512]", arg538_1: "f32[128]", arg539_1: "f32[512, 128]", arg540_1: "f32[512]", arg541_1: "f32[128, 512]", arg542_1: "f32[128]", arg543_1: "f32[128, 512]", arg544_1: "f32[128]", arg545_1: "f32[128, 128]", arg546_1: "f32[128]", arg547_1: "f32[128, 128]", arg548_1: "f32[128]", arg549_1: "f32[128, 512]", arg550_1: "f32[128]", arg551_1: "f32[128, 128]", arg552_1: "f32[128]", arg553_1: "f32[512, 128]", arg554_1: "f32[512]", arg555_1: "f32[128, 512]", arg556_1: "f32[128]", arg557_1: "f32[512, 128]", arg558_1: "f32[512]", arg559_1: "f32[128, 512]", arg560_1: "f32[128]", arg561_1: "f32[512, 128]", arg562_1: "f32[512]", arg563_1: "f32[128, 512]", arg564_1: "f32[128]", arg565_1: "f32[512, 128]", arg566_1: "f32[512]", arg567_1: "f32[128, 512]", arg568_1: "f32[128]", arg569_1: "f32[512, 128]", arg570_1: "f32[512]", arg571_1: "f32[128, 512]", arg572_1: "f32[128]", arg573_1: "f32[128, 512]", arg574_1: "f32[128]", arg575_1: "f32[128, 128]", arg576_1: "f32[128]", arg577_1: "f32[128, 128]", arg578_1: "f32[128]", arg579_1: "f32[128, 512]", arg580_1: "f32[128]", arg581_1: "f32[128, 128]", arg582_1: "f32[128]", arg583_1: "f32[512, 128]", arg584_1: "f32[512]", arg585_1: "f32[128, 512]", arg586_1: "f32[128]", arg587_1: "f32[512, 128]", arg588_1: "f32[512]", arg589_1: "f32[128, 512]", arg590_1: "f32[128]", arg591_1: "f32[512, 128]", arg592_1: "f32[512]", arg593_1: "f32[128, 512]", arg594_1: "f32[128]", arg595_1: "f32[512, 128]", arg596_1: "f32[512]", arg597_1: "f32[128, 512]", arg598_1: "f32[128]", arg599_1: "f32[512, 128]", arg600_1: "f32[512]", arg601_1: "f32[128, 512]", arg602_1: "f32[128]", arg603_1: "f32[128, 512]", arg604_1: "f32[128]", arg605_1: "f32[128, 128]", arg606_1: "f32[128]", arg607_1: "f32[128, 128]", arg608_1: "f32[128]", arg609_1: "f32[128, 512]", arg610_1: "f32[128]", arg611_1: "f32[128, 128]", arg612_1: "f32[128]", arg613_1: "f32[512, 128]", arg614_1: "f32[512]", arg615_1: "f32[128, 512]", arg616_1: "f32[128]", arg617_1: "f32[512, 128]", arg618_1: "f32[512]", arg619_1: "f32[128, 512]", arg620_1: "f32[128]", arg621_1: "f32[512, 128]", arg622_1: "f32[512]", arg623_1: "f32[128, 512]", arg624_1: "f32[128]", arg625_1: "f32[512, 128]", arg626_1: "f32[512]", arg627_1: "f32[128, 512]", arg628_1: "f32[128]", arg629_1: "f32[512, 128]", arg630_1: "f32[512]", arg631_1: "f32[128, 512]", arg632_1: "f32[128]", arg633_1: "f32[128, 512]", arg634_1: "f32[128]", arg635_1: "f32[128, 128]", arg636_1: "f32[128]", arg637_1: "f32[128, 128]", arg638_1: "f32[128]", arg639_1: "f32[128, 512]", arg640_1: "f32[128]", arg641_1: "f32[128, 128]", arg642_1: "f32[128]", arg643_1: "f32[512, 128]", arg644_1: "f32[512]", arg645_1: "f32[128, 512]", arg646_1: "f32[128]", arg647_1: "f32[512, 128]", arg648_1: "f32[512]", arg649_1: "f32[128, 512]", arg650_1: "f32[128]", arg651_1: "f32[512, 128]", arg652_1: "f32[512]", arg653_1: "f32[128, 512]", arg654_1: "f32[128]", arg655_1: "f32[512, 128]", arg656_1: "f32[512]", arg657_1: "f32[128, 512]", arg658_1: "f32[128]", arg659_1: "f32[512, 128]", arg660_1: "f32[512]", arg661_1: "f32[128, 512]", arg662_1: "f32[128]", arg663_1: "f32[128, 512]", arg664_1: "f32[128]", arg665_1: "f32[128, 128]", arg666_1: "f32[128]", arg667_1: "f32[128, 128]", arg668_1: "f32[128]", arg669_1: "f32[128, 512]", arg670_1: "f32[128]", arg671_1: "f32[128, 128]", arg672_1: "f32[128]", arg673_1: "f32[512, 128]", arg674_1: "f32[512]", arg675_1: "f32[128, 512]", arg676_1: "f32[128]", arg677_1: "f32[512, 128]", arg678_1: "f32[512]", arg679_1: "f32[128, 512]", arg680_1: "f32[128]", arg681_1: "f32[512, 128]", arg682_1: "f32[512]", arg683_1: "f32[128, 512]", arg684_1: "f32[128]", arg685_1: "f32[512, 128]", arg686_1: "f32[512]", arg687_1: "f32[128, 512]", arg688_1: "f32[128]", arg689_1: "f32[512, 128]", arg690_1: "f32[512]", arg691_1: "f32[128, 512]", arg692_1: "f32[128]", arg693_1: "f32[128, 512]", arg694_1: "f32[128]", arg695_1: "f32[128, 128]", arg696_1: "f32[128]", arg697_1: "f32[128, 128]", arg698_1: "f32[128]", arg699_1: "f32[128, 512]", arg700_1: "f32[128]", arg701_1: "f32[128, 128]", arg702_1: "f32[128]", arg703_1: "f32[512, 128]", arg704_1: "f32[512]", arg705_1: "f32[128, 512]", arg706_1: "f32[128]", arg707_1: "f32[512, 128]", arg708_1: "f32[512]", arg709_1: "f32[128, 512]", arg710_1: "f32[128]", arg711_1: "f32[512, 128]", arg712_1: "f32[512]", arg713_1: "f32[128, 512]", arg714_1: "f32[128]", arg715_1: "f32[512, 128]", arg716_1: "f32[512]", arg717_1: "f32[128, 512]", arg718_1: "f32[128]", arg719_1: "f32[512, 128]", arg720_1: "f32[512]", arg721_1: "f32[128, 512]", arg722_1: "f32[128]", arg723_1: "f32[128, 512]", arg724_1: "f32[128]", arg725_1: "f32[128, 128]", arg726_1: "f32[128]", arg727_1: "f32[128, 128]", arg728_1: "f32[128]", arg729_1: "f32[128, 512]", arg730_1: "f32[128]", arg731_1: "f32[128, 128]", arg732_1: "f32[128]", arg733_1: "f32[512, 128]", arg734_1: "f32[512]", arg735_1: "f32[128, 512]", arg736_1: "f32[128]", arg737_1: "f32[512, 128]", arg738_1: "f32[512]", arg739_1: "f32[128, 512]", arg740_1: "f32[128]", arg741_1: "f32[512, 128]", arg742_1: "f32[512]", arg743_1: "f32[128, 512]", arg744_1: "f32[128]", arg745_1: "f32[512, 128]", arg746_1: "f32[512]", arg747_1: "f32[128, 512]", arg748_1: "f32[128]", arg749_1: "f32[512, 128]", arg750_1: "f32[512]", arg751_1: "f32[128, 512]", arg752_1: "f32[128]", arg753_1: "f32[128, 512]", arg754_1: "f32[128]", arg755_1: "f32[128, 128]", arg756_1: "f32[128]", arg757_1: "f32[128, 128]", arg758_1: "f32[128]", arg759_1: "f32[128, 512]", arg760_1: "f32[128]", arg761_1: "f32[128, 128]", arg762_1: "f32[128]", arg763_1: "f32[512, 128]", arg764_1: "f32[512]", arg765_1: "f32[128, 512]", arg766_1: "f32[128]", arg767_1: "f32[512, 128]", arg768_1: "f32[512]", arg769_1: "f32[128, 512]", arg770_1: "f32[128]", arg771_1: "f32[512, 128]", arg772_1: "f32[512]", arg773_1: "f32[128, 512]", arg774_1: "f32[128]", arg775_1: "f32[512, 128]", arg776_1: "f32[512]", arg777_1: "f32[128, 512]", arg778_1: "f32[128]", arg779_1: "f32[512, 128]", arg780_1: "f32[512]", arg781_1: "f32[128, 512]", arg782_1: "f32[128]", arg783_1: "f32[128, 512]", arg784_1: "f32[128]", arg785_1: "f32[128, 128]", arg786_1: "f32[128]", arg787_1: "f32[128, 128]", arg788_1: "f32[128]", arg789_1: "f32[128, 512]", arg790_1: "f32[128]", arg791_1: "f32[128, 128]", arg792_1: "f32[128]", arg793_1: "f32[512, 128]", arg794_1: "f32[512]", arg795_1: "f32[128, 512]", arg796_1: "f32[128]", arg797_1: "f32[512, 128]", arg798_1: "f32[512]", arg799_1: "f32[128, 512]", arg800_1: "f32[128]", arg801_1: "f32[512, 128]", arg802_1: "f32[512]", arg803_1: "f32[128, 512]", arg804_1: "f32[128]", arg805_1: "f32[512, 128]", arg806_1: "f32[512]", arg807_1: "f32[128, 512]", arg808_1: "f32[128]", arg809_1: "f32[512, 128]", arg810_1: "f32[512]", arg811_1: "f32[128, 512]", arg812_1: "f32[128]", arg813_1: "f32[128, 512]", arg814_1: "f32[128]", arg815_1: "f32[128, 128]", arg816_1: "f32[128]", arg817_1: "f32[128, 128]", arg818_1: "f32[128]", arg819_1: "f32[128, 512]", arg820_1: "f32[128]", arg821_1: "f32[128, 128]", arg822_1: "f32[128]", arg823_1: "f32[512, 128]", arg824_1: "f32[512]", arg825_1: "f32[128, 512]", arg826_1: "f32[128]", arg827_1: "f32[512, 128]", arg828_1: "f32[512]", arg829_1: "f32[128, 512]", arg830_1: "f32[128]", arg831_1: "f32[512, 128]", arg832_1: "f32[512]", arg833_1: "f32[128, 512]", arg834_1: "f32[128]", arg835_1: "f32[512, 128]", arg836_1: "f32[512]", arg837_1: "f32[128, 512]", arg838_1: "f32[128]", arg839_1: "f32[512, 128]", arg840_1: "f32[512]", arg841_1: "f32[128, 512]", arg842_1: "f32[128]", arg843_1: "f32[128, 512]", arg844_1: "f32[128]", arg845_1: "f32[128, 128]", arg846_1: "f32[128]", arg847_1: "f32[128, 128]", arg848_1: "f32[128]", arg849_1: "f32[128, 512]", arg850_1: "f32[128]", arg851_1: "f32[128, 128]", arg852_1: "f32[128]", arg853_1: "f32[512, 128]", arg854_1: "f32[512]", arg855_1: "f32[128, 512]", arg856_1: "f32[128]", arg857_1: "f32[512, 128]", arg858_1: "f32[512]", arg859_1: "f32[128, 512]", arg860_1: "f32[128]", arg861_1: "f32[512, 128]", arg862_1: "f32[512]", arg863_1: "f32[128, 512]", arg864_1: "f32[128]", arg865_1: "f32[512, 128]", arg866_1: "f32[512]", arg867_1: "f32[128, 512]", arg868_1: "f32[128]", arg869_1: "f32[512, 128]", arg870_1: "f32[512]", arg871_1: "f32[128, 512]", arg872_1: "f32[128]", arg873_1: "f32[128, 512]", arg874_1: "f32[128]", arg875_1: "f32[128, 128]", arg876_1: "f32[128]", arg877_1: "f32[128, 128]", arg878_1: "f32[128]", arg879_1: "f32[128, 512]", arg880_1: "f32[128]", arg881_1: "f32[128, 128]", arg882_1: "f32[128]", arg883_1: "f32[512, 128]", arg884_1: "f32[512]", arg885_1: "f32[128, 512]", arg886_1: "f32[128]", arg887_1: "f32[512, 128]", arg888_1: "f32[512]", arg889_1: "f32[128, 512]", arg890_1: "f32[128]", arg891_1: "f32[512, 128]", arg892_1: "f32[512]", arg893_1: "f32[128, 512]", arg894_1: "f32[128]", arg895_1: "f32[512, 128]", arg896_1: "f32[512]", arg897_1: "f32[128, 512]", arg898_1: "f32[128]", arg899_1: "f32[512, 128]", arg900_1: "f32[512]", arg901_1: "f32[128, 512]", arg902_1: "f32[128]", arg903_1: "f32[128, 512]", arg904_1: "f32[128]", arg905_1: "f32[128, 128]", arg906_1: "f32[128]", arg907_1: "f32[128, 128]", arg908_1: "f32[128]", arg909_1: "f32[128, 512]", arg910_1: "f32[128]", arg911_1: "f32[128, 128]", arg912_1: "f32[128]", arg913_1: "f32[512, 128]", arg914_1: "f32[512]", arg915_1: "f32[128, 512]", arg916_1: "f32[128]", arg917_1: "f32[512, 128]", arg918_1: "f32[512]", arg919_1: "f32[128, 512]", arg920_1: "f32[128]", arg921_1: "f32[512, 128]", arg922_1: "f32[512]", arg923_1: "f32[128, 512]", arg924_1: "f32[128]", arg925_1: "f32[512, 128]", arg926_1: "f32[512]", arg927_1: "f32[128, 512]", arg928_1: "f32[128]", arg929_1: "f32[512, 128]", arg930_1: "f32[512]", arg931_1: "f32[128, 512]", arg932_1: "f32[128]", arg933_1: "f32[128, 512]", arg934_1: "f32[128]", arg935_1: "f32[128, 128]", arg936_1: "f32[128]", arg937_1: "f32[128, 128]", arg938_1: "f32[128]", arg939_1: "f32[128, 512]", arg940_1: "f32[128]", arg941_1: "f32[128, 128]", arg942_1: "f32[128]", arg943_1: "f32[512, 128]", arg944_1: "f32[512]", arg945_1: "f32[128, 512]", arg946_1: "f32[128]", arg947_1: "f32[512, 128]", arg948_1: "f32[512]", arg949_1: "f32[128, 512]", arg950_1: "f32[128]", arg951_1: "f32[512, 128]", arg952_1: "f32[512]", arg953_1: "f32[128, 512]", arg954_1: "f32[128]", arg955_1: "f32[512, 128]", arg956_1: "f32[512]", arg957_1: "f32[128, 512]", arg958_1: "f32[128]", arg959_1: "f32[512, 128]", arg960_1: "f32[512]", arg961_1: "f32[128, 512]", arg962_1: "f32[128]", arg963_1: "f32[128, 512]", arg964_1: "f32[128]", arg965_1: "f32[128, 128]", arg966_1: "f32[128]", arg967_1: "f32[128, 128]", arg968_1: "f32[128]", arg969_1: "f32[128, 512]", arg970_1: "f32[128]", arg971_1: "f32[128, 128]", arg972_1: "f32[128]", arg973_1: "f32[512, 128]", arg974_1: "f32[512]", arg975_1: "f32[128, 512]", arg976_1: "f32[128]", arg977_1: "f32[512, 128]", arg978_1: "f32[512]", arg979_1: "f32[128, 512]", arg980_1: "f32[128]", arg981_1: "f32[512, 128]", arg982_1: "f32[512]", arg983_1: "f32[128, 512]", arg984_1: "f32[128]", arg985_1: "f32[512, 128]", arg986_1: "f32[512]", arg987_1: "f32[128, 512]", arg988_1: "f32[128]", arg989_1: "f32[512, 128]", arg990_1: "f32[512]", arg991_1: "f32[128, 512]", arg992_1: "f32[128]", arg993_1: "f32[128, 512]", arg994_1: "f32[128]", arg995_1: "f32[128, 128]", arg996_1: "f32[128]", arg997_1: "f32[128, 128]", arg998_1: "f32[128]", arg999_1: "f32[128, 512]", arg1000_1: "f32[128]", arg1001_1: "f32[128, 128]", arg1002_1: "f32[128]", arg1003_1: "f32[512, 128]", arg1004_1: "f32[512]", arg1005_1: "f32[128, 512]", arg1006_1: "f32[128]", arg1007_1: "f32[512, 128]", arg1008_1: "f32[512]", arg1009_1: "f32[128, 512]", arg1010_1: "f32[128]", arg1011_1: "f32[512, 128]", arg1012_1: "f32[512]", arg1013_1: "f32[128, 512]", arg1014_1: "f32[128]", arg1015_1: "f32[512, 128]", arg1016_1: "f32[512]", arg1017_1: "f32[128, 512]", arg1018_1: "f32[128]", arg1019_1: "f32[512, 128]", arg1020_1: "f32[512]", arg1021_1: "f32[128, 512]", arg1022_1: "f32[128]", arg1023_1: "f32[128, 512]", arg1024_1: "f32[128]", arg1025_1: "f32[128, 128]", arg1026_1: "f32[128]", arg1027_1: "f32[128, 128]", arg1028_1: "f32[128]", arg1029_1: "f32[128, 512]", arg1030_1: "f32[128]", arg1031_1: "f32[128, 128]", arg1032_1: "f32[128]", arg1033_1: "f32[512, 128]", arg1034_1: "f32[512]", arg1035_1: "f32[128, 512]", arg1036_1: "f32[128]", arg1037_1: "f32[512, 128]", arg1038_1: "f32[512]", arg1039_1: "f32[128, 512]", arg1040_1: "f32[128]", arg1041_1: "f32[512, 128]", arg1042_1: "f32[512]", arg1043_1: "f32[128, 512]", arg1044_1: "f32[128]", arg1045_1: "f32[512, 128]", arg1046_1: "f32[512]", arg1047_1: "f32[128, 512]", arg1048_1: "f32[128]", arg1049_1: "f32[512, 128]", arg1050_1: "f32[512]", arg1051_1: "f32[128, 512]", arg1052_1: "f32[128]", arg1053_1: "f32[128, 512]", arg1054_1: "f32[128]", arg1055_1: "f32[128, 128]", arg1056_1: "f32[128]", arg1057_1: "f32[128, 128]", arg1058_1: "f32[128]", arg1059_1: "f32[128, 512]", arg1060_1: "f32[128]", arg1061_1: "f32[128, 128]", arg1062_1: "f32[128]", arg1063_1: "f32[512, 128]", arg1064_1: "f32[512]", arg1065_1: "f32[128, 512]", arg1066_1: "f32[128]", arg1067_1: "f32[512, 128]", arg1068_1: "f32[512]", arg1069_1: "f32[128, 512]", arg1070_1: "f32[128]", arg1071_1: "f32[512, 128]", arg1072_1: "f32[512]", arg1073_1: "f32[128, 512]", arg1074_1: "f32[128]", arg1075_1: "f32[512, 128]", arg1076_1: "f32[512]", arg1077_1: "f32[128, 512]", arg1078_1: "f32[128]", arg1079_1: "f32[512, 128]", arg1080_1: "f32[512]", arg1081_1: "f32[128, 512]", arg1082_1: "f32[128]", arg1083_1: "f32[128, 512]", arg1084_1: "f32[128]", arg1085_1: "f32[128, 128]", arg1086_1: "f32[128]", arg1087_1: "f32[128, 128]", arg1088_1: "f32[128]", arg1089_1: "f32[128, 512]", arg1090_1: "f32[128]", arg1091_1: "f32[128, 128]", arg1092_1: "f32[128]", arg1093_1: "f32[512, 128]", arg1094_1: "f32[512]", arg1095_1: "f32[128, 512]", arg1096_1: "f32[128]", arg1097_1: "f32[512, 128]", arg1098_1: "f32[512]", arg1099_1: "f32[128, 512]", arg1100_1: "f32[128]", arg1101_1: "f32[512, 128]", arg1102_1: "f32[512]", arg1103_1: "f32[128, 512]", arg1104_1: "f32[128]", arg1105_1: "f32[512, 128]", arg1106_1: "f32[512]", arg1107_1: "f32[128, 512]", arg1108_1: "f32[128]", arg1109_1: "f32[512, 128]", arg1110_1: "f32[512]", arg1111_1: "f32[2, 512]", arg1112_1: "f32[2]", arg1113_1: "i64[1, 512]", arg1114_1: "i64[1, 128]", arg1115_1: "i64[1]", arg1116_1: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:880, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    unsqueeze: "f32[1, 1, 128]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 128]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    full_default_1: "f32[1, 1, 1, 128]" = torch.ops.aten.full.default([1, 1, 1, 128], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:218, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 128, 128]" = torch.ops.aten.embedding.default(arg386_1, arg1114_1, 0);  arg386_1 = arg1114_1 = None
    
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
    permute: "f32[384, 512]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
    addmm: "f32[128, 512]" = torch.ops.aten.addmm.default(arg388_1, view, permute);  arg388_1 = view = permute = None
    view_1: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm, [1, 128, 512]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:213, code: position_ids = self.position_ids[:, :seq_length]
    slice_4: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1113_1, 1, 0, 128);  arg1113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:241, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg389_1, slice_4);  arg389_1 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:243, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:882, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_default: "i64[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:242, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_2: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg390_1, full_default);  arg390_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:243, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add_1: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
    add_2: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_4: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_2: "f32[512, 128]" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
    addmm_2: "f32[128, 128]" = torch.ops.aten.addmm.default(arg394_1, view_4, permute_2);  arg394_1 = view_4 = permute_2 = None
    view_5: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 128]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_3: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_5, arg4_1);  view_5 = arg4_1 = None
    add_4: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_6: "f32[128, 128]" = torch.ops.aten.reshape.default(add_4, [128, 128])
    permute_3: "f32[128, 128]" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
    addmm_3: "f32[128, 128]" = torch.ops.aten.addmm.default(arg396_1, view_6, permute_3);  arg396_1 = view_6 = permute_3 = None
    view_7: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_12: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_7, [1, 128, 4, 32]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    
    # No stacktrace found for following nodes
    clone_default_69: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_8: "f32[128, 128]" = torch.ops.aten.reshape.default(add_4, [128, 128]);  add_4 = None
    permute_4: "f32[128, 128]" = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
    addmm_4: "f32[128, 128]" = torch.ops.aten.addmm.default(arg398_1, view_8, permute_4);  arg398_1 = view_8 = permute_4 = None
    view_9: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 128]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_13: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_9, [1, 128, 4, 32]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_7: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # No stacktrace found for following nodes
    clone_default_70: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_10: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_5: "f32[512, 128]" = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
    addmm_5: "f32[128, 128]" = torch.ops.aten.addmm.default(arg400_1, view_10, permute_5);  arg400_1 = view_10 = permute_5 = None
    view_11: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_14: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_11, [1, 128, 4, 32]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_8: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # No stacktrace found for following nodes
    clone_default_71: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    _scaled_dot_product_flash_attention_default_23 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_69, clone_default_70, clone_default_71, scale = 0.17677669529663687);  clone_default_69 = clone_default_70 = clone_default_71 = None
    getitem_25: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_23[0];  _scaled_dot_product_flash_attention_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_10: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_25, [0, 2, 1, 3]);  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_21: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_10, [1, 128, 128]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_22: "f32[128, 128]" = torch.ops.aten.reshape.default(view_21, [128, 128]);  view_21 = None
    permute_11: "f32[128, 128]" = torch.ops.aten.permute.default(arg401_1, [1, 0]);  arg401_1 = None
    addmm_6: "f32[128, 128]" = torch.ops.aten.addmm.default(arg402_1, view_22, permute_11);  arg402_1 = view_22 = permute_11 = None
    view_23: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 128]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_2: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_1: "f32[512, 128]" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
    addmm_1: "f32[128, 128]" = torch.ops.aten.addmm.default(arg392_1, view_2, permute_1);  arg392_1 = view_2 = permute_1 = None
    view_3: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 128]);  addmm_1 = None
    
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
    permute_12: "f32[128, 512]" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
    addmm_7: "f32[128, 512]" = torch.ops.aten.addmm.default(arg404_1, view_24, permute_12);  arg404_1 = view_24 = permute_12 = None
    view_25: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 512]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_25);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_26: "f32[128, 512]" = torch.ops.aten.reshape.default(relu, [128, 512]);  relu = None
    permute_13: "f32[512, 128]" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
    addmm_8: "f32[128, 128]" = torch.ops.aten.addmm.default(arg406_1, view_26, permute_13);  arg406_1 = view_26 = permute_13 = None
    view_27: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 128]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_8: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_27, add_7);  view_27 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_5: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_8, arg8_1);  add_8 = arg8_1 = None
    add_9: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_5, arg9_1);  mul_5 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_28: "f32[128, 128]" = torch.ops.aten.reshape.default(add_9, [128, 128])
    permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
    addmm_9: "f32[128, 512]" = torch.ops.aten.addmm.default(arg408_1, view_28, permute_14);  arg408_1 = view_28 = permute_14 = None
    view_29: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 512]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_1: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_29);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_30: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_1, [128, 512]);  relu_1 = None
    permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
    addmm_10: "f32[128, 128]" = torch.ops.aten.addmm.default(arg410_1, view_30, permute_15);  arg410_1 = view_30 = permute_15 = None
    view_31: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 128]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_10: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_31, add_9);  view_31 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_6: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_10, arg10_1);  add_10 = arg10_1 = None
    add_11: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, arg11_1);  mul_6 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[128, 128]" = torch.ops.aten.reshape.default(add_11, [128, 128])
    permute_16: "f32[128, 512]" = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
    addmm_11: "f32[128, 512]" = torch.ops.aten.addmm.default(arg412_1, view_32, permute_16);  arg412_1 = view_32 = permute_16 = None
    view_33: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 512]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_2: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_34: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_2, [128, 512]);  relu_2 = None
    permute_17: "f32[512, 128]" = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
    addmm_12: "f32[128, 128]" = torch.ops.aten.addmm.default(arg414_1, view_34, permute_17);  arg414_1 = view_34 = permute_17 = None
    view_35: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 128]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_12: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_35, add_11);  view_35 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_7: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_12, arg12_1);  add_12 = arg12_1 = None
    add_13: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_7, arg13_1);  mul_7 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_36: "f32[128, 128]" = torch.ops.aten.reshape.default(add_13, [128, 128])
    permute_18: "f32[128, 512]" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
    addmm_13: "f32[128, 512]" = torch.ops.aten.addmm.default(arg416_1, view_36, permute_18);  arg416_1 = view_36 = permute_18 = None
    view_37: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_3: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_37);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_38: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_3, [128, 512]);  relu_3 = None
    permute_19: "f32[512, 128]" = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
    addmm_14: "f32[128, 128]" = torch.ops.aten.addmm.default(arg418_1, view_38, permute_19);  arg418_1 = view_38 = permute_19 = None
    view_39: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 128]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_14: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_39, add_13);  view_39 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_8: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, arg14_1);  add_14 = arg14_1 = None
    add_15: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_40: "f32[128, 128]" = torch.ops.aten.reshape.default(add_15, [128, 128]);  add_15 = None
    permute_20: "f32[128, 512]" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
    addmm_15: "f32[128, 512]" = torch.ops.aten.addmm.default(arg420_1, view_40, permute_20);  arg420_1 = view_40 = permute_20 = None
    view_41: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_16: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_41, add_2);  view_41 = add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_16, arg16_1);  add_16 = arg16_1 = None
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_9, arg17_1);  mul_9 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_44: "f32[128, 512]" = torch.ops.aten.reshape.default(add_17, [128, 512])
    permute_22: "f32[512, 128]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
    addmm_17: "f32[128, 128]" = torch.ops.aten.addmm.default(arg424_1, view_44, permute_22);  arg424_1 = view_44 = permute_22 = None
    view_45: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 128]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_11: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_45, arg20_1);  view_45 = arg20_1 = None
    add_19: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_46: "f32[128, 128]" = torch.ops.aten.reshape.default(add_19, [128, 128])
    permute_23: "f32[128, 128]" = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
    addmm_18: "f32[128, 128]" = torch.ops.aten.addmm.default(arg426_1, view_46, permute_23);  arg426_1 = view_46 = permute_23 = None
    view_47: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 128]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_52: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_47, [1, 128, 4, 32]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # No stacktrace found for following nodes
    clone_default_66: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_48: "f32[128, 128]" = torch.ops.aten.reshape.default(add_19, [128, 128]);  add_19 = None
    permute_24: "f32[128, 128]" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
    addmm_19: "f32[128, 128]" = torch.ops.aten.addmm.default(arg428_1, view_48, permute_24);  arg428_1 = view_48 = permute_24 = None
    view_49: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 128]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_53: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_49, [1, 128, 4, 32]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # No stacktrace found for following nodes
    clone_default_67: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_50: "f32[128, 512]" = torch.ops.aten.reshape.default(add_17, [128, 512])
    permute_25: "f32[512, 128]" = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
    addmm_20: "f32[128, 128]" = torch.ops.aten.addmm.default(arg430_1, view_50, permute_25);  arg430_1 = view_50 = permute_25 = None
    view_51: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 128]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_54: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_51, [1, 128, 4, 32]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # No stacktrace found for following nodes
    clone_default_68: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    _scaled_dot_product_flash_attention_default_22 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_66, clone_default_67, clone_default_68, scale = 0.17677669529663687);  clone_default_66 = clone_default_67 = clone_default_68 = None
    getitem_24: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_22[0];  _scaled_dot_product_flash_attention_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_30: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_24, [0, 2, 1, 3]);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_61: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_30, [1, 128, 128]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_62: "f32[128, 128]" = torch.ops.aten.reshape.default(view_61, [128, 128]);  view_61 = None
    permute_31: "f32[128, 128]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
    addmm_21: "f32[128, 128]" = torch.ops.aten.addmm.default(arg432_1, view_62, permute_31);  arg432_1 = view_62 = permute_31 = None
    view_63: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 128]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_42: "f32[128, 512]" = torch.ops.aten.reshape.default(add_17, [128, 512])
    permute_21: "f32[512, 128]" = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
    addmm_16: "f32[128, 128]" = torch.ops.aten.addmm.default(arg422_1, view_42, permute_21);  arg422_1 = view_42 = permute_21 = None
    view_43: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 128]);  addmm_16 = None
    
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
    permute_32: "f32[128, 512]" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
    addmm_22: "f32[128, 512]" = torch.ops.aten.addmm.default(arg434_1, view_64, permute_32);  arg434_1 = view_64 = permute_32 = None
    view_65: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 512]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_4: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_66: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_4, [128, 512]);  relu_4 = None
    permute_33: "f32[512, 128]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
    addmm_23: "f32[128, 128]" = torch.ops.aten.addmm.default(arg436_1, view_66, permute_33);  arg436_1 = view_66 = permute_33 = None
    view_67: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 128]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_23: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_67, add_22);  view_67 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_13: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_23, arg24_1);  add_23 = arg24_1 = None
    add_24: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, arg25_1);  mul_13 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[128, 128]" = torch.ops.aten.reshape.default(add_24, [128, 128])
    permute_34: "f32[128, 512]" = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
    addmm_24: "f32[128, 512]" = torch.ops.aten.addmm.default(arg438_1, view_68, permute_34);  arg438_1 = view_68 = permute_34 = None
    view_69: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 512]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_5: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_69);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_70: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_5, [128, 512]);  relu_5 = None
    permute_35: "f32[512, 128]" = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
    addmm_25: "f32[128, 128]" = torch.ops.aten.addmm.default(arg440_1, view_70, permute_35);  arg440_1 = view_70 = permute_35 = None
    view_71: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 128]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_25: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_71, add_24);  view_71 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_14: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, arg26_1);  add_25 = arg26_1 = None
    add_26: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, arg27_1);  mul_14 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_72: "f32[128, 128]" = torch.ops.aten.reshape.default(add_26, [128, 128])
    permute_36: "f32[128, 512]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
    addmm_26: "f32[128, 512]" = torch.ops.aten.addmm.default(arg442_1, view_72, permute_36);  arg442_1 = view_72 = permute_36 = None
    view_73: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 512]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_6: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_74: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_6, [128, 512]);  relu_6 = None
    permute_37: "f32[512, 128]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
    addmm_27: "f32[128, 128]" = torch.ops.aten.addmm.default(arg444_1, view_74, permute_37);  arg444_1 = view_74 = permute_37 = None
    view_75: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 128]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_27: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_75, add_26);  view_75 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_15: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_27, arg28_1);  add_27 = arg28_1 = None
    add_28: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_15, arg29_1);  mul_15 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_76: "f32[128, 128]" = torch.ops.aten.reshape.default(add_28, [128, 128])
    permute_38: "f32[128, 512]" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
    addmm_28: "f32[128, 512]" = torch.ops.aten.addmm.default(arg446_1, view_76, permute_38);  arg446_1 = view_76 = permute_38 = None
    view_77: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 512]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_7: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_77);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_78: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_7, [128, 512]);  relu_7 = None
    permute_39: "f32[512, 128]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
    addmm_29: "f32[128, 128]" = torch.ops.aten.addmm.default(arg448_1, view_78, permute_39);  arg448_1 = view_78 = permute_39 = None
    view_79: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 128]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_29: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_79, add_28);  view_79 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_16: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_29, arg30_1);  add_29 = arg30_1 = None
    add_30: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_16, arg31_1);  mul_16 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_80: "f32[128, 128]" = torch.ops.aten.reshape.default(add_30, [128, 128]);  add_30 = None
    permute_40: "f32[128, 512]" = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
    addmm_30: "f32[128, 512]" = torch.ops.aten.addmm.default(arg450_1, view_80, permute_40);  arg450_1 = view_80 = permute_40 = None
    view_81: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 512]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_31: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_81, add_17);  view_81 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_17: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_31, arg32_1);  add_31 = arg32_1 = None
    add_32: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_17, arg33_1);  mul_17 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_84: "f32[128, 512]" = torch.ops.aten.reshape.default(add_32, [128, 512])
    permute_42: "f32[512, 128]" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
    addmm_32: "f32[128, 128]" = torch.ops.aten.addmm.default(arg454_1, view_84, permute_42);  arg454_1 = view_84 = permute_42 = None
    view_85: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 128]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_19: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_85, arg36_1);  view_85 = arg36_1 = None
    add_34: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, arg37_1);  mul_19 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_86: "f32[128, 128]" = torch.ops.aten.reshape.default(add_34, [128, 128])
    permute_43: "f32[128, 128]" = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
    addmm_33: "f32[128, 128]" = torch.ops.aten.addmm.default(arg456_1, view_86, permute_43);  arg456_1 = view_86 = permute_43 = None
    view_87: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 128]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_92: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_87, [1, 128, 4, 32]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # No stacktrace found for following nodes
    clone_default_63: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_88: "f32[128, 128]" = torch.ops.aten.reshape.default(add_34, [128, 128]);  add_34 = None
    permute_44: "f32[128, 128]" = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
    addmm_34: "f32[128, 128]" = torch.ops.aten.addmm.default(arg458_1, view_88, permute_44);  arg458_1 = view_88 = permute_44 = None
    view_89: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 128]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_93: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_89, [1, 128, 4, 32]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # No stacktrace found for following nodes
    clone_default_64: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_90: "f32[128, 512]" = torch.ops.aten.reshape.default(add_32, [128, 512])
    permute_45: "f32[512, 128]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
    addmm_35: "f32[128, 128]" = torch.ops.aten.addmm.default(arg460_1, view_90, permute_45);  arg460_1 = view_90 = permute_45 = None
    view_91: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 128]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_94: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_91, [1, 128, 4, 32]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # No stacktrace found for following nodes
    clone_default_65: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    _scaled_dot_product_flash_attention_default_21 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_63, clone_default_64, clone_default_65, scale = 0.17677669529663687);  clone_default_63 = clone_default_64 = clone_default_65 = None
    getitem_23: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_21[0];  _scaled_dot_product_flash_attention_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_50: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_23, [0, 2, 1, 3]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_101: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_50, [1, 128, 128]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_102: "f32[128, 128]" = torch.ops.aten.reshape.default(view_101, [128, 128]);  view_101 = None
    permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
    addmm_36: "f32[128, 128]" = torch.ops.aten.addmm.default(arg462_1, view_102, permute_51);  arg462_1 = view_102 = permute_51 = None
    view_103: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 128]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_82: "f32[128, 512]" = torch.ops.aten.reshape.default(add_32, [128, 512])
    permute_41: "f32[512, 128]" = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
    addmm_31: "f32[128, 128]" = torch.ops.aten.addmm.default(arg452_1, view_82, permute_41);  arg452_1 = view_82 = permute_41 = None
    view_83: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 128]);  addmm_31 = None
    
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
    permute_52: "f32[128, 512]" = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
    addmm_37: "f32[128, 512]" = torch.ops.aten.addmm.default(arg464_1, view_104, permute_52);  arg464_1 = view_104 = permute_52 = None
    view_105: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_8: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_106: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_8, [128, 512]);  relu_8 = None
    permute_53: "f32[512, 128]" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
    addmm_38: "f32[128, 128]" = torch.ops.aten.addmm.default(arg466_1, view_106, permute_53);  arg466_1 = view_106 = permute_53 = None
    view_107: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 128]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_38: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_107, add_37);  view_107 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_21: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_38, arg40_1);  add_38 = arg40_1 = None
    add_39: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, arg41_1);  mul_21 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[128, 128]" = torch.ops.aten.reshape.default(add_39, [128, 128])
    permute_54: "f32[128, 512]" = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
    addmm_39: "f32[128, 512]" = torch.ops.aten.addmm.default(arg468_1, view_108, permute_54);  arg468_1 = view_108 = permute_54 = None
    view_109: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_9: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_110: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_9, [128, 512]);  relu_9 = None
    permute_55: "f32[512, 128]" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
    addmm_40: "f32[128, 128]" = torch.ops.aten.addmm.default(arg470_1, view_110, permute_55);  arg470_1 = view_110 = permute_55 = None
    view_111: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 128]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_40: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_111, add_39);  view_111 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_22: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_40, arg42_1);  add_40 = arg42_1 = None
    add_41: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, arg43_1);  mul_22 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_112: "f32[128, 128]" = torch.ops.aten.reshape.default(add_41, [128, 128])
    permute_56: "f32[128, 512]" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
    addmm_41: "f32[128, 512]" = torch.ops.aten.addmm.default(arg472_1, view_112, permute_56);  arg472_1 = view_112 = permute_56 = None
    view_113: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_10: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_114: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_10, [128, 512]);  relu_10 = None
    permute_57: "f32[512, 128]" = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
    addmm_42: "f32[128, 128]" = torch.ops.aten.addmm.default(arg474_1, view_114, permute_57);  arg474_1 = view_114 = permute_57 = None
    view_115: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 128]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_42: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_115, add_41);  view_115 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_23: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_42, arg44_1);  add_42 = arg44_1 = None
    add_43: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_23, arg45_1);  mul_23 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_116: "f32[128, 128]" = torch.ops.aten.reshape.default(add_43, [128, 128])
    permute_58: "f32[128, 512]" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
    addmm_43: "f32[128, 512]" = torch.ops.aten.addmm.default(arg476_1, view_116, permute_58);  arg476_1 = view_116 = permute_58 = None
    view_117: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_11: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_117);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_118: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_11, [128, 512]);  relu_11 = None
    permute_59: "f32[512, 128]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
    addmm_44: "f32[128, 128]" = torch.ops.aten.addmm.default(arg478_1, view_118, permute_59);  arg478_1 = view_118 = permute_59 = None
    view_119: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 128]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_44: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_119, add_43);  view_119 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_24: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_44, arg46_1);  add_44 = arg46_1 = None
    add_45: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_24, arg47_1);  mul_24 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_120: "f32[128, 128]" = torch.ops.aten.reshape.default(add_45, [128, 128]);  add_45 = None
    permute_60: "f32[128, 512]" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
    addmm_45: "f32[128, 512]" = torch.ops.aten.addmm.default(arg480_1, view_120, permute_60);  arg480_1 = view_120 = permute_60 = None
    view_121: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_46: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_121, add_32);  view_121 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_25: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_46, arg48_1);  add_46 = arg48_1 = None
    add_47: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_25, arg49_1);  mul_25 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_124: "f32[128, 512]" = torch.ops.aten.reshape.default(add_47, [128, 512])
    permute_62: "f32[512, 128]" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
    addmm_47: "f32[128, 128]" = torch.ops.aten.addmm.default(arg484_1, view_124, permute_62);  arg484_1 = view_124 = permute_62 = None
    view_125: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 128]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_27: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_125, arg52_1);  view_125 = arg52_1 = None
    add_49: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_27, arg53_1);  mul_27 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_126: "f32[128, 128]" = torch.ops.aten.reshape.default(add_49, [128, 128])
    permute_63: "f32[128, 128]" = torch.ops.aten.permute.default(arg485_1, [1, 0]);  arg485_1 = None
    addmm_48: "f32[128, 128]" = torch.ops.aten.addmm.default(arg486_1, view_126, permute_63);  arg486_1 = view_126 = permute_63 = None
    view_127: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 128]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_132: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_127, [1, 128, 4, 32]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_66: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # No stacktrace found for following nodes
    clone_default_60: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_128: "f32[128, 128]" = torch.ops.aten.reshape.default(add_49, [128, 128]);  add_49 = None
    permute_64: "f32[128, 128]" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
    addmm_49: "f32[128, 128]" = torch.ops.aten.addmm.default(arg488_1, view_128, permute_64);  arg488_1 = view_128 = permute_64 = None
    view_129: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 128]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_133: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_129, [1, 128, 4, 32]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_67: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # No stacktrace found for following nodes
    clone_default_61: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_130: "f32[128, 512]" = torch.ops.aten.reshape.default(add_47, [128, 512])
    permute_65: "f32[512, 128]" = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
    addmm_50: "f32[128, 128]" = torch.ops.aten.addmm.default(arg490_1, view_130, permute_65);  arg490_1 = view_130 = permute_65 = None
    view_131: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 128]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_134: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_131, [1, 128, 4, 32]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    
    # No stacktrace found for following nodes
    clone_default_62: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    _scaled_dot_product_flash_attention_default_20 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_60, clone_default_61, clone_default_62, scale = 0.17677669529663687);  clone_default_60 = clone_default_61 = clone_default_62 = None
    getitem_22: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_20[0];  _scaled_dot_product_flash_attention_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_70: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_22, [0, 2, 1, 3]);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_141: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_70, [1, 128, 128]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_142: "f32[128, 128]" = torch.ops.aten.reshape.default(view_141, [128, 128]);  view_141 = None
    permute_71: "f32[128, 128]" = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
    addmm_51: "f32[128, 128]" = torch.ops.aten.addmm.default(arg492_1, view_142, permute_71);  arg492_1 = view_142 = permute_71 = None
    view_143: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 128]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_122: "f32[128, 512]" = torch.ops.aten.reshape.default(add_47, [128, 512])
    permute_61: "f32[512, 128]" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
    addmm_46: "f32[128, 128]" = torch.ops.aten.addmm.default(arg482_1, view_122, permute_61);  arg482_1 = view_122 = permute_61 = None
    view_123: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 128]);  addmm_46 = None
    
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
    permute_72: "f32[128, 512]" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
    addmm_52: "f32[128, 512]" = torch.ops.aten.addmm.default(arg494_1, view_144, permute_72);  arg494_1 = view_144 = permute_72 = None
    view_145: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 512]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_12: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_145);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_146: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_12, [128, 512]);  relu_12 = None
    permute_73: "f32[512, 128]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
    addmm_53: "f32[128, 128]" = torch.ops.aten.addmm.default(arg496_1, view_146, permute_73);  arg496_1 = view_146 = permute_73 = None
    view_147: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 128]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_53: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_147, add_52);  view_147 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_29: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_53, arg56_1);  add_53 = arg56_1 = None
    add_54: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_29, arg57_1);  mul_29 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[128, 128]" = torch.ops.aten.reshape.default(add_54, [128, 128])
    permute_74: "f32[128, 512]" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
    addmm_54: "f32[128, 512]" = torch.ops.aten.addmm.default(arg498_1, view_148, permute_74);  arg498_1 = view_148 = permute_74 = None
    view_149: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 512]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_13: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_149);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_150: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_13, [128, 512]);  relu_13 = None
    permute_75: "f32[512, 128]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
    addmm_55: "f32[128, 128]" = torch.ops.aten.addmm.default(arg500_1, view_150, permute_75);  arg500_1 = view_150 = permute_75 = None
    view_151: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 128]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_55: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_151, add_54);  view_151 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_30: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_55, arg58_1);  add_55 = arg58_1 = None
    add_56: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, arg59_1);  mul_30 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[128, 128]" = torch.ops.aten.reshape.default(add_56, [128, 128])
    permute_76: "f32[128, 512]" = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
    addmm_56: "f32[128, 512]" = torch.ops.aten.addmm.default(arg502_1, view_152, permute_76);  arg502_1 = view_152 = permute_76 = None
    view_153: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 512]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_14: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_154: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_14, [128, 512]);  relu_14 = None
    permute_77: "f32[512, 128]" = torch.ops.aten.permute.default(arg503_1, [1, 0]);  arg503_1 = None
    addmm_57: "f32[128, 128]" = torch.ops.aten.addmm.default(arg504_1, view_154, permute_77);  arg504_1 = view_154 = permute_77 = None
    view_155: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_57, [1, 128, 128]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_57: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_155, add_56);  view_155 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_31: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_57, arg60_1);  add_57 = arg60_1 = None
    add_58: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_31, arg61_1);  mul_31 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_156: "f32[128, 128]" = torch.ops.aten.reshape.default(add_58, [128, 128])
    permute_78: "f32[128, 512]" = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
    addmm_58: "f32[128, 512]" = torch.ops.aten.addmm.default(arg506_1, view_156, permute_78);  arg506_1 = view_156 = permute_78 = None
    view_157: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_58, [1, 128, 512]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_15: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_157);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_158: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_15, [128, 512]);  relu_15 = None
    permute_79: "f32[512, 128]" = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
    addmm_59: "f32[128, 128]" = torch.ops.aten.addmm.default(arg508_1, view_158, permute_79);  arg508_1 = view_158 = permute_79 = None
    view_159: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_59, [1, 128, 128]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_59: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_159, add_58);  view_159 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_32: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_59, arg62_1);  add_59 = arg62_1 = None
    add_60: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_32, arg63_1);  mul_32 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_160: "f32[128, 128]" = torch.ops.aten.reshape.default(add_60, [128, 128]);  add_60 = None
    permute_80: "f32[128, 512]" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
    addmm_60: "f32[128, 512]" = torch.ops.aten.addmm.default(arg510_1, view_160, permute_80);  arg510_1 = view_160 = permute_80 = None
    view_161: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_60, [1, 128, 512]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_61: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_161, add_47);  view_161 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_61, arg64_1);  add_61 = arg64_1 = None
    add_62: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_33, arg65_1);  mul_33 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_164: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_82: "f32[512, 128]" = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
    addmm_62: "f32[128, 128]" = torch.ops.aten.addmm.default(arg514_1, view_164, permute_82);  arg514_1 = view_164 = permute_82 = None
    view_165: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_62, [1, 128, 128]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_35: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_165, arg68_1);  view_165 = arg68_1 = None
    add_64: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_166: "f32[128, 128]" = torch.ops.aten.reshape.default(add_64, [128, 128])
    permute_83: "f32[128, 128]" = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
    addmm_63: "f32[128, 128]" = torch.ops.aten.addmm.default(arg516_1, view_166, permute_83);  arg516_1 = view_166 = permute_83 = None
    view_167: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_63, [1, 128, 128]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_172: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_167, [1, 128, 4, 32]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_86: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # No stacktrace found for following nodes
    clone_default_57: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_168: "f32[128, 128]" = torch.ops.aten.reshape.default(add_64, [128, 128]);  add_64 = None
    permute_84: "f32[128, 128]" = torch.ops.aten.permute.default(arg517_1, [1, 0]);  arg517_1 = None
    addmm_64: "f32[128, 128]" = torch.ops.aten.addmm.default(arg518_1, view_168, permute_84);  arg518_1 = view_168 = permute_84 = None
    view_169: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_64, [1, 128, 128]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_173: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_169, [1, 128, 4, 32]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_87: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    
    # No stacktrace found for following nodes
    clone_default_58: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_170: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_85: "f32[512, 128]" = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
    addmm_65: "f32[128, 128]" = torch.ops.aten.addmm.default(arg520_1, view_170, permute_85);  arg520_1 = view_170 = permute_85 = None
    view_171: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_65, [1, 128, 128]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_174: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_171, [1, 128, 4, 32]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_88: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # No stacktrace found for following nodes
    clone_default_59: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    _scaled_dot_product_flash_attention_default_19 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_57, clone_default_58, clone_default_59, scale = 0.17677669529663687);  clone_default_57 = clone_default_58 = clone_default_59 = None
    getitem_21: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_19[0];  _scaled_dot_product_flash_attention_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_90: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_181: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_90, [1, 128, 128]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_182: "f32[128, 128]" = torch.ops.aten.reshape.default(view_181, [128, 128]);  view_181 = None
    permute_91: "f32[128, 128]" = torch.ops.aten.permute.default(arg521_1, [1, 0]);  arg521_1 = None
    addmm_66: "f32[128, 128]" = torch.ops.aten.addmm.default(arg522_1, view_182, permute_91);  arg522_1 = view_182 = permute_91 = None
    view_183: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_66, [1, 128, 128]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_162: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_81: "f32[512, 128]" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
    addmm_61: "f32[128, 128]" = torch.ops.aten.addmm.default(arg512_1, view_162, permute_81);  arg512_1 = view_162 = permute_81 = None
    view_163: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_61, [1, 128, 128]);  addmm_61 = None
    
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
    permute_92: "f32[128, 512]" = torch.ops.aten.permute.default(arg523_1, [1, 0]);  arg523_1 = None
    addmm_67: "f32[128, 512]" = torch.ops.aten.addmm.default(arg524_1, view_184, permute_92);  arg524_1 = view_184 = permute_92 = None
    view_185: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_67, [1, 128, 512]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_16: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_185);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_186: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_16, [128, 512]);  relu_16 = None
    permute_93: "f32[512, 128]" = torch.ops.aten.permute.default(arg525_1, [1, 0]);  arg525_1 = None
    addmm_68: "f32[128, 128]" = torch.ops.aten.addmm.default(arg526_1, view_186, permute_93);  arg526_1 = view_186 = permute_93 = None
    view_187: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_68, [1, 128, 128]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_68: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_187, add_67);  view_187 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_37: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_68, arg72_1);  add_68 = arg72_1 = None
    add_69: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, arg73_1);  mul_37 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_188: "f32[128, 128]" = torch.ops.aten.reshape.default(add_69, [128, 128])
    permute_94: "f32[128, 512]" = torch.ops.aten.permute.default(arg527_1, [1, 0]);  arg527_1 = None
    addmm_69: "f32[128, 512]" = torch.ops.aten.addmm.default(arg528_1, view_188, permute_94);  arg528_1 = view_188 = permute_94 = None
    view_189: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_69, [1, 128, 512]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_17: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_190: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_17, [128, 512]);  relu_17 = None
    permute_95: "f32[512, 128]" = torch.ops.aten.permute.default(arg529_1, [1, 0]);  arg529_1 = None
    addmm_70: "f32[128, 128]" = torch.ops.aten.addmm.default(arg530_1, view_190, permute_95);  arg530_1 = view_190 = permute_95 = None
    view_191: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_70, [1, 128, 128]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_70: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_191, add_69);  view_191 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_38: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_70, arg74_1);  add_70 = arg74_1 = None
    add_71: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, arg75_1);  mul_38 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[128, 128]" = torch.ops.aten.reshape.default(add_71, [128, 128])
    permute_96: "f32[128, 512]" = torch.ops.aten.permute.default(arg531_1, [1, 0]);  arg531_1 = None
    addmm_71: "f32[128, 512]" = torch.ops.aten.addmm.default(arg532_1, view_192, permute_96);  arg532_1 = view_192 = permute_96 = None
    view_193: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_71, [1, 128, 512]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_18: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_193);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_194: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_18, [128, 512]);  relu_18 = None
    permute_97: "f32[512, 128]" = torch.ops.aten.permute.default(arg533_1, [1, 0]);  arg533_1 = None
    addmm_72: "f32[128, 128]" = torch.ops.aten.addmm.default(arg534_1, view_194, permute_97);  arg534_1 = view_194 = permute_97 = None
    view_195: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_72, [1, 128, 128]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_72: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_195, add_71);  view_195 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_39: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_72, arg76_1);  add_72 = arg76_1 = None
    add_73: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_39, arg77_1);  mul_39 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[128, 128]" = torch.ops.aten.reshape.default(add_73, [128, 128])
    permute_98: "f32[128, 512]" = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
    addmm_73: "f32[128, 512]" = torch.ops.aten.addmm.default(arg536_1, view_196, permute_98);  arg536_1 = view_196 = permute_98 = None
    view_197: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_73, [1, 128, 512]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_19: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_197);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_198: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_19, [128, 512]);  relu_19 = None
    permute_99: "f32[512, 128]" = torch.ops.aten.permute.default(arg537_1, [1, 0]);  arg537_1 = None
    addmm_74: "f32[128, 128]" = torch.ops.aten.addmm.default(arg538_1, view_198, permute_99);  arg538_1 = view_198 = permute_99 = None
    view_199: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_74, [1, 128, 128]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_74: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_199, add_73);  view_199 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_40: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_74, arg78_1);  add_74 = arg78_1 = None
    add_75: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_40, arg79_1);  mul_40 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_200: "f32[128, 128]" = torch.ops.aten.reshape.default(add_75, [128, 128]);  add_75 = None
    permute_100: "f32[128, 512]" = torch.ops.aten.permute.default(arg539_1, [1, 0]);  arg539_1 = None
    addmm_75: "f32[128, 512]" = torch.ops.aten.addmm.default(arg540_1, view_200, permute_100);  arg540_1 = view_200 = permute_100 = None
    view_201: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_75, [1, 128, 512]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_76: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_201, add_62);  view_201 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_76, arg80_1);  add_76 = arg80_1 = None
    add_77: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_41, arg81_1);  mul_41 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_204: "f32[128, 512]" = torch.ops.aten.reshape.default(add_77, [128, 512])
    permute_102: "f32[512, 128]" = torch.ops.aten.permute.default(arg543_1, [1, 0]);  arg543_1 = None
    addmm_77: "f32[128, 128]" = torch.ops.aten.addmm.default(arg544_1, view_204, permute_102);  arg544_1 = view_204 = permute_102 = None
    view_205: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_77, [1, 128, 128]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_43: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_205, arg84_1);  view_205 = arg84_1 = None
    add_79: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_43, arg85_1);  mul_43 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_206: "f32[128, 128]" = torch.ops.aten.reshape.default(add_79, [128, 128])
    permute_103: "f32[128, 128]" = torch.ops.aten.permute.default(arg545_1, [1, 0]);  arg545_1 = None
    addmm_78: "f32[128, 128]" = torch.ops.aten.addmm.default(arg546_1, view_206, permute_103);  arg546_1 = view_206 = permute_103 = None
    view_207: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_78, [1, 128, 128]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_212: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_207, [1, 128, 4, 32]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_106: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # No stacktrace found for following nodes
    clone_default_54: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_208: "f32[128, 128]" = torch.ops.aten.reshape.default(add_79, [128, 128]);  add_79 = None
    permute_104: "f32[128, 128]" = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
    addmm_79: "f32[128, 128]" = torch.ops.aten.addmm.default(arg548_1, view_208, permute_104);  arg548_1 = view_208 = permute_104 = None
    view_209: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_79, [1, 128, 128]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_213: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_209, [1, 128, 4, 32]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_107: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # No stacktrace found for following nodes
    clone_default_55: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_210: "f32[128, 512]" = torch.ops.aten.reshape.default(add_77, [128, 512])
    permute_105: "f32[512, 128]" = torch.ops.aten.permute.default(arg549_1, [1, 0]);  arg549_1 = None
    addmm_80: "f32[128, 128]" = torch.ops.aten.addmm.default(arg550_1, view_210, permute_105);  arg550_1 = view_210 = permute_105 = None
    view_211: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_80, [1, 128, 128]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_214: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_211, [1, 128, 4, 32]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_108: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    
    # No stacktrace found for following nodes
    clone_default_56: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    _scaled_dot_product_flash_attention_default_18 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_54, clone_default_55, clone_default_56, scale = 0.17677669529663687);  clone_default_54 = clone_default_55 = clone_default_56 = None
    getitem_20: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_18[0];  _scaled_dot_product_flash_attention_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_110: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_221: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_110, [1, 128, 128]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_222: "f32[128, 128]" = torch.ops.aten.reshape.default(view_221, [128, 128]);  view_221 = None
    permute_111: "f32[128, 128]" = torch.ops.aten.permute.default(arg551_1, [1, 0]);  arg551_1 = None
    addmm_81: "f32[128, 128]" = torch.ops.aten.addmm.default(arg552_1, view_222, permute_111);  arg552_1 = view_222 = permute_111 = None
    view_223: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_81, [1, 128, 128]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_202: "f32[128, 512]" = torch.ops.aten.reshape.default(add_77, [128, 512])
    permute_101: "f32[512, 128]" = torch.ops.aten.permute.default(arg541_1, [1, 0]);  arg541_1 = None
    addmm_76: "f32[128, 128]" = torch.ops.aten.addmm.default(arg542_1, view_202, permute_101);  arg542_1 = view_202 = permute_101 = None
    view_203: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_76, [1, 128, 128]);  addmm_76 = None
    
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
    permute_112: "f32[128, 512]" = torch.ops.aten.permute.default(arg553_1, [1, 0]);  arg553_1 = None
    addmm_82: "f32[128, 512]" = torch.ops.aten.addmm.default(arg554_1, view_224, permute_112);  arg554_1 = view_224 = permute_112 = None
    view_225: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_82, [1, 128, 512]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_20: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_225);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_226: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_20, [128, 512]);  relu_20 = None
    permute_113: "f32[512, 128]" = torch.ops.aten.permute.default(arg555_1, [1, 0]);  arg555_1 = None
    addmm_83: "f32[128, 128]" = torch.ops.aten.addmm.default(arg556_1, view_226, permute_113);  arg556_1 = view_226 = permute_113 = None
    view_227: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_83, [1, 128, 128]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_83: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_227, add_82);  view_227 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_45: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_83, arg88_1);  add_83 = arg88_1 = None
    add_84: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_45, arg89_1);  mul_45 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_228: "f32[128, 128]" = torch.ops.aten.reshape.default(add_84, [128, 128])
    permute_114: "f32[128, 512]" = torch.ops.aten.permute.default(arg557_1, [1, 0]);  arg557_1 = None
    addmm_84: "f32[128, 512]" = torch.ops.aten.addmm.default(arg558_1, view_228, permute_114);  arg558_1 = view_228 = permute_114 = None
    view_229: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_84, [1, 128, 512]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_21: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_229);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_230: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_21, [128, 512]);  relu_21 = None
    permute_115: "f32[512, 128]" = torch.ops.aten.permute.default(arg559_1, [1, 0]);  arg559_1 = None
    addmm_85: "f32[128, 128]" = torch.ops.aten.addmm.default(arg560_1, view_230, permute_115);  arg560_1 = view_230 = permute_115 = None
    view_231: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_85, [1, 128, 128]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_85: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_231, add_84);  view_231 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_46: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_85, arg90_1);  add_85 = arg90_1 = None
    add_86: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, arg91_1);  mul_46 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_232: "f32[128, 128]" = torch.ops.aten.reshape.default(add_86, [128, 128])
    permute_116: "f32[128, 512]" = torch.ops.aten.permute.default(arg561_1, [1, 0]);  arg561_1 = None
    addmm_86: "f32[128, 512]" = torch.ops.aten.addmm.default(arg562_1, view_232, permute_116);  arg562_1 = view_232 = permute_116 = None
    view_233: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_86, [1, 128, 512]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_22: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_233);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_234: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_22, [128, 512]);  relu_22 = None
    permute_117: "f32[512, 128]" = torch.ops.aten.permute.default(arg563_1, [1, 0]);  arg563_1 = None
    addmm_87: "f32[128, 128]" = torch.ops.aten.addmm.default(arg564_1, view_234, permute_117);  arg564_1 = view_234 = permute_117 = None
    view_235: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_87, [1, 128, 128]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_87: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_235, add_86);  view_235 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_47: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_87, arg92_1);  add_87 = arg92_1 = None
    add_88: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_47, arg93_1);  mul_47 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[128, 128]" = torch.ops.aten.reshape.default(add_88, [128, 128])
    permute_118: "f32[128, 512]" = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
    addmm_88: "f32[128, 512]" = torch.ops.aten.addmm.default(arg566_1, view_236, permute_118);  arg566_1 = view_236 = permute_118 = None
    view_237: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_88, [1, 128, 512]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_23: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_237);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_238: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_23, [128, 512]);  relu_23 = None
    permute_119: "f32[512, 128]" = torch.ops.aten.permute.default(arg567_1, [1, 0]);  arg567_1 = None
    addmm_89: "f32[128, 128]" = torch.ops.aten.addmm.default(arg568_1, view_238, permute_119);  arg568_1 = view_238 = permute_119 = None
    view_239: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_89, [1, 128, 128]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_89: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_239, add_88);  view_239 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_48: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_89, arg94_1);  add_89 = arg94_1 = None
    add_90: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_48, arg95_1);  mul_48 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_240: "f32[128, 128]" = torch.ops.aten.reshape.default(add_90, [128, 128]);  add_90 = None
    permute_120: "f32[128, 512]" = torch.ops.aten.permute.default(arg569_1, [1, 0]);  arg569_1 = None
    addmm_90: "f32[128, 512]" = torch.ops.aten.addmm.default(arg570_1, view_240, permute_120);  arg570_1 = view_240 = permute_120 = None
    view_241: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_90, [1, 128, 512]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_91: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_241, add_77);  view_241 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_49: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_91, arg96_1);  add_91 = arg96_1 = None
    add_92: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_49, arg97_1);  mul_49 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_244: "f32[128, 512]" = torch.ops.aten.reshape.default(add_92, [128, 512])
    permute_122: "f32[512, 128]" = torch.ops.aten.permute.default(arg573_1, [1, 0]);  arg573_1 = None
    addmm_92: "f32[128, 128]" = torch.ops.aten.addmm.default(arg574_1, view_244, permute_122);  arg574_1 = view_244 = permute_122 = None
    view_245: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_92, [1, 128, 128]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_51: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_245, arg100_1);  view_245 = arg100_1 = None
    add_94: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_51, arg101_1);  mul_51 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_246: "f32[128, 128]" = torch.ops.aten.reshape.default(add_94, [128, 128])
    permute_123: "f32[128, 128]" = torch.ops.aten.permute.default(arg575_1, [1, 0]);  arg575_1 = None
    addmm_93: "f32[128, 128]" = torch.ops.aten.addmm.default(arg576_1, view_246, permute_123);  arg576_1 = view_246 = permute_123 = None
    view_247: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_93, [1, 128, 128]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_252: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_247, [1, 128, 4, 32]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # No stacktrace found for following nodes
    clone_default_51: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_248: "f32[128, 128]" = torch.ops.aten.reshape.default(add_94, [128, 128]);  add_94 = None
    permute_124: "f32[128, 128]" = torch.ops.aten.permute.default(arg577_1, [1, 0]);  arg577_1 = None
    addmm_94: "f32[128, 128]" = torch.ops.aten.addmm.default(arg578_1, view_248, permute_124);  arg578_1 = view_248 = permute_124 = None
    view_249: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_94, [1, 128, 128]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_253: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_249, [1, 128, 4, 32]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    
    # No stacktrace found for following nodes
    clone_default_52: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_250: "f32[128, 512]" = torch.ops.aten.reshape.default(add_92, [128, 512])
    permute_125: "f32[512, 128]" = torch.ops.aten.permute.default(arg579_1, [1, 0]);  arg579_1 = None
    addmm_95: "f32[128, 128]" = torch.ops.aten.addmm.default(arg580_1, view_250, permute_125);  arg580_1 = view_250 = permute_125 = None
    view_251: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_95, [1, 128, 128]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_254: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_251, [1, 128, 4, 32]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_128: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
    
    # No stacktrace found for following nodes
    clone_default_53: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    _scaled_dot_product_flash_attention_default_17 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_51, clone_default_52, clone_default_53, scale = 0.17677669529663687);  clone_default_51 = clone_default_52 = clone_default_53 = None
    getitem_19: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_17[0];  _scaled_dot_product_flash_attention_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_130: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3]);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_261: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_130, [1, 128, 128]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_262: "f32[128, 128]" = torch.ops.aten.reshape.default(view_261, [128, 128]);  view_261 = None
    permute_131: "f32[128, 128]" = torch.ops.aten.permute.default(arg581_1, [1, 0]);  arg581_1 = None
    addmm_96: "f32[128, 128]" = torch.ops.aten.addmm.default(arg582_1, view_262, permute_131);  arg582_1 = view_262 = permute_131 = None
    view_263: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_96, [1, 128, 128]);  addmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_242: "f32[128, 512]" = torch.ops.aten.reshape.default(add_92, [128, 512])
    permute_121: "f32[512, 128]" = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
    addmm_91: "f32[128, 128]" = torch.ops.aten.addmm.default(arg572_1, view_242, permute_121);  arg572_1 = view_242 = permute_121 = None
    view_243: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_91, [1, 128, 128]);  addmm_91 = None
    
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
    permute_132: "f32[128, 512]" = torch.ops.aten.permute.default(arg583_1, [1, 0]);  arg583_1 = None
    addmm_97: "f32[128, 512]" = torch.ops.aten.addmm.default(arg584_1, view_264, permute_132);  arg584_1 = view_264 = permute_132 = None
    view_265: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_97, [1, 128, 512]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_24: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_266: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_24, [128, 512]);  relu_24 = None
    permute_133: "f32[512, 128]" = torch.ops.aten.permute.default(arg585_1, [1, 0]);  arg585_1 = None
    addmm_98: "f32[128, 128]" = torch.ops.aten.addmm.default(arg586_1, view_266, permute_133);  arg586_1 = view_266 = permute_133 = None
    view_267: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_98, [1, 128, 128]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_98: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_97);  view_267 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_53: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_98, arg104_1);  add_98 = arg104_1 = None
    add_99: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_53, arg105_1);  mul_53 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_268: "f32[128, 128]" = torch.ops.aten.reshape.default(add_99, [128, 128])
    permute_134: "f32[128, 512]" = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
    addmm_99: "f32[128, 512]" = torch.ops.aten.addmm.default(arg588_1, view_268, permute_134);  arg588_1 = view_268 = permute_134 = None
    view_269: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_99, [1, 128, 512]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_25: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_269);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_270: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_25, [128, 512]);  relu_25 = None
    permute_135: "f32[512, 128]" = torch.ops.aten.permute.default(arg589_1, [1, 0]);  arg589_1 = None
    addmm_100: "f32[128, 128]" = torch.ops.aten.addmm.default(arg590_1, view_270, permute_135);  arg590_1 = view_270 = permute_135 = None
    view_271: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_100, [1, 128, 128]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_100: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_271, add_99);  view_271 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_54: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_100, arg106_1);  add_100 = arg106_1 = None
    add_101: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_54, arg107_1);  mul_54 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_272: "f32[128, 128]" = torch.ops.aten.reshape.default(add_101, [128, 128])
    permute_136: "f32[128, 512]" = torch.ops.aten.permute.default(arg591_1, [1, 0]);  arg591_1 = None
    addmm_101: "f32[128, 512]" = torch.ops.aten.addmm.default(arg592_1, view_272, permute_136);  arg592_1 = view_272 = permute_136 = None
    view_273: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_101, [1, 128, 512]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_26: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_273);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_274: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_26, [128, 512]);  relu_26 = None
    permute_137: "f32[512, 128]" = torch.ops.aten.permute.default(arg593_1, [1, 0]);  arg593_1 = None
    addmm_102: "f32[128, 128]" = torch.ops.aten.addmm.default(arg594_1, view_274, permute_137);  arg594_1 = view_274 = permute_137 = None
    view_275: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_102, [1, 128, 128]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_102: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_275, add_101);  view_275 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_55: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_102, arg108_1);  add_102 = arg108_1 = None
    add_103: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_55, arg109_1);  mul_55 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_276: "f32[128, 128]" = torch.ops.aten.reshape.default(add_103, [128, 128])
    permute_138: "f32[128, 512]" = torch.ops.aten.permute.default(arg595_1, [1, 0]);  arg595_1 = None
    addmm_103: "f32[128, 512]" = torch.ops.aten.addmm.default(arg596_1, view_276, permute_138);  arg596_1 = view_276 = permute_138 = None
    view_277: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_103, [1, 128, 512]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_27: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_277);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_278: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_27, [128, 512]);  relu_27 = None
    permute_139: "f32[512, 128]" = torch.ops.aten.permute.default(arg597_1, [1, 0]);  arg597_1 = None
    addmm_104: "f32[128, 128]" = torch.ops.aten.addmm.default(arg598_1, view_278, permute_139);  arg598_1 = view_278 = permute_139 = None
    view_279: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_104, [1, 128, 128]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_104: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_279, add_103);  view_279 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_56: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_104, arg110_1);  add_104 = arg110_1 = None
    add_105: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_56, arg111_1);  mul_56 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_280: "f32[128, 128]" = torch.ops.aten.reshape.default(add_105, [128, 128]);  add_105 = None
    permute_140: "f32[128, 512]" = torch.ops.aten.permute.default(arg599_1, [1, 0]);  arg599_1 = None
    addmm_105: "f32[128, 512]" = torch.ops.aten.addmm.default(arg600_1, view_280, permute_140);  arg600_1 = view_280 = permute_140 = None
    view_281: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_105, [1, 128, 512]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_106: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_281, add_92);  view_281 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_106, arg112_1);  add_106 = arg112_1 = None
    add_107: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_57, arg113_1);  mul_57 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_284: "f32[128, 512]" = torch.ops.aten.reshape.default(add_107, [128, 512])
    permute_142: "f32[512, 128]" = torch.ops.aten.permute.default(arg603_1, [1, 0]);  arg603_1 = None
    addmm_107: "f32[128, 128]" = torch.ops.aten.addmm.default(arg604_1, view_284, permute_142);  arg604_1 = view_284 = permute_142 = None
    view_285: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_107, [1, 128, 128]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_59: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_285, arg116_1);  view_285 = arg116_1 = None
    add_109: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, arg117_1);  mul_59 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_286: "f32[128, 128]" = torch.ops.aten.reshape.default(add_109, [128, 128])
    permute_143: "f32[128, 128]" = torch.ops.aten.permute.default(arg605_1, [1, 0]);  arg605_1 = None
    addmm_108: "f32[128, 128]" = torch.ops.aten.addmm.default(arg606_1, view_286, permute_143);  arg606_1 = view_286 = permute_143 = None
    view_287: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_108, [1, 128, 128]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_292: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_287, [1, 128, 4, 32]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_146: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # No stacktrace found for following nodes
    clone_default_48: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_288: "f32[128, 128]" = torch.ops.aten.reshape.default(add_109, [128, 128]);  add_109 = None
    permute_144: "f32[128, 128]" = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
    addmm_109: "f32[128, 128]" = torch.ops.aten.addmm.default(arg608_1, view_288, permute_144);  arg608_1 = view_288 = permute_144 = None
    view_289: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_109, [1, 128, 128]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_293: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_289, [1, 128, 4, 32]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_147: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    
    # No stacktrace found for following nodes
    clone_default_49: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_290: "f32[128, 512]" = torch.ops.aten.reshape.default(add_107, [128, 512])
    permute_145: "f32[512, 128]" = torch.ops.aten.permute.default(arg609_1, [1, 0]);  arg609_1 = None
    addmm_110: "f32[128, 128]" = torch.ops.aten.addmm.default(arg610_1, view_290, permute_145);  arg610_1 = view_290 = permute_145 = None
    view_291: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_110, [1, 128, 128]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_294: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_291, [1, 128, 4, 32]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_148: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # No stacktrace found for following nodes
    clone_default_50: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    _scaled_dot_product_flash_attention_default_16 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_48, clone_default_49, clone_default_50, scale = 0.17677669529663687);  clone_default_48 = clone_default_49 = clone_default_50 = None
    getitem_18: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_16[0];  _scaled_dot_product_flash_attention_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_301: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_150, [1, 128, 128]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_302: "f32[128, 128]" = torch.ops.aten.reshape.default(view_301, [128, 128]);  view_301 = None
    permute_151: "f32[128, 128]" = torch.ops.aten.permute.default(arg611_1, [1, 0]);  arg611_1 = None
    addmm_111: "f32[128, 128]" = torch.ops.aten.addmm.default(arg612_1, view_302, permute_151);  arg612_1 = view_302 = permute_151 = None
    view_303: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_111, [1, 128, 128]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_282: "f32[128, 512]" = torch.ops.aten.reshape.default(add_107, [128, 512])
    permute_141: "f32[512, 128]" = torch.ops.aten.permute.default(arg601_1, [1, 0]);  arg601_1 = None
    addmm_106: "f32[128, 128]" = torch.ops.aten.addmm.default(arg602_1, view_282, permute_141);  arg602_1 = view_282 = permute_141 = None
    view_283: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_106, [1, 128, 128]);  addmm_106 = None
    
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
    permute_152: "f32[128, 512]" = torch.ops.aten.permute.default(arg613_1, [1, 0]);  arg613_1 = None
    addmm_112: "f32[128, 512]" = torch.ops.aten.addmm.default(arg614_1, view_304, permute_152);  arg614_1 = view_304 = permute_152 = None
    view_305: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_112, [1, 128, 512]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_28: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_305);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_306: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_28, [128, 512]);  relu_28 = None
    permute_153: "f32[512, 128]" = torch.ops.aten.permute.default(arg615_1, [1, 0]);  arg615_1 = None
    addmm_113: "f32[128, 128]" = torch.ops.aten.addmm.default(arg616_1, view_306, permute_153);  arg616_1 = view_306 = permute_153 = None
    view_307: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_113, [1, 128, 128]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_113: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_307, add_112);  view_307 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_61: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_113, arg120_1);  add_113 = arg120_1 = None
    add_114: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, arg121_1);  mul_61 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_308: "f32[128, 128]" = torch.ops.aten.reshape.default(add_114, [128, 128])
    permute_154: "f32[128, 512]" = torch.ops.aten.permute.default(arg617_1, [1, 0]);  arg617_1 = None
    addmm_114: "f32[128, 512]" = torch.ops.aten.addmm.default(arg618_1, view_308, permute_154);  arg618_1 = view_308 = permute_154 = None
    view_309: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_114, [1, 128, 512]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_29: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_310: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_29, [128, 512]);  relu_29 = None
    permute_155: "f32[512, 128]" = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
    addmm_115: "f32[128, 128]" = torch.ops.aten.addmm.default(arg620_1, view_310, permute_155);  arg620_1 = view_310 = permute_155 = None
    view_311: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_115, [1, 128, 128]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_115: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_311, add_114);  view_311 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_62: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_115, arg122_1);  add_115 = arg122_1 = None
    add_116: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_62, arg123_1);  mul_62 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_312: "f32[128, 128]" = torch.ops.aten.reshape.default(add_116, [128, 128])
    permute_156: "f32[128, 512]" = torch.ops.aten.permute.default(arg621_1, [1, 0]);  arg621_1 = None
    addmm_116: "f32[128, 512]" = torch.ops.aten.addmm.default(arg622_1, view_312, permute_156);  arg622_1 = view_312 = permute_156 = None
    view_313: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_116, [1, 128, 512]);  addmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_30: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_313);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_314: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_30, [128, 512]);  relu_30 = None
    permute_157: "f32[512, 128]" = torch.ops.aten.permute.default(arg623_1, [1, 0]);  arg623_1 = None
    addmm_117: "f32[128, 128]" = torch.ops.aten.addmm.default(arg624_1, view_314, permute_157);  arg624_1 = view_314 = permute_157 = None
    view_315: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_117, [1, 128, 128]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_117: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_315, add_116);  view_315 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_63: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_117, arg124_1);  add_117 = arg124_1 = None
    add_118: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_63, arg125_1);  mul_63 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_316: "f32[128, 128]" = torch.ops.aten.reshape.default(add_118, [128, 128])
    permute_158: "f32[128, 512]" = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
    addmm_118: "f32[128, 512]" = torch.ops.aten.addmm.default(arg626_1, view_316, permute_158);  arg626_1 = view_316 = permute_158 = None
    view_317: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_118, [1, 128, 512]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_31: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_317);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_318: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_31, [128, 512]);  relu_31 = None
    permute_159: "f32[512, 128]" = torch.ops.aten.permute.default(arg627_1, [1, 0]);  arg627_1 = None
    addmm_119: "f32[128, 128]" = torch.ops.aten.addmm.default(arg628_1, view_318, permute_159);  arg628_1 = view_318 = permute_159 = None
    view_319: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_119, [1, 128, 128]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_119: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_319, add_118);  view_319 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_64: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, arg126_1);  add_119 = arg126_1 = None
    add_120: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_64, arg127_1);  mul_64 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_320: "f32[128, 128]" = torch.ops.aten.reshape.default(add_120, [128, 128]);  add_120 = None
    permute_160: "f32[128, 512]" = torch.ops.aten.permute.default(arg629_1, [1, 0]);  arg629_1 = None
    addmm_120: "f32[128, 512]" = torch.ops.aten.addmm.default(arg630_1, view_320, permute_160);  arg630_1 = view_320 = permute_160 = None
    view_321: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_120, [1, 128, 512]);  addmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_121: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_321, add_107);  view_321 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_121, arg128_1);  add_121 = arg128_1 = None
    add_122: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_65, arg129_1);  mul_65 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_324: "f32[128, 512]" = torch.ops.aten.reshape.default(add_122, [128, 512])
    permute_162: "f32[512, 128]" = torch.ops.aten.permute.default(arg633_1, [1, 0]);  arg633_1 = None
    addmm_122: "f32[128, 128]" = torch.ops.aten.addmm.default(arg634_1, view_324, permute_162);  arg634_1 = view_324 = permute_162 = None
    view_325: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_122, [1, 128, 128]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_67: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_325, arg132_1);  view_325 = arg132_1 = None
    add_124: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_67, arg133_1);  mul_67 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_326: "f32[128, 128]" = torch.ops.aten.reshape.default(add_124, [128, 128])
    permute_163: "f32[128, 128]" = torch.ops.aten.permute.default(arg635_1, [1, 0]);  arg635_1 = None
    addmm_123: "f32[128, 128]" = torch.ops.aten.addmm.default(arg636_1, view_326, permute_163);  arg636_1 = view_326 = permute_163 = None
    view_327: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_123, [1, 128, 128]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_332: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_327, [1, 128, 4, 32]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_166: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    
    # No stacktrace found for following nodes
    clone_default_45: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_328: "f32[128, 128]" = torch.ops.aten.reshape.default(add_124, [128, 128]);  add_124 = None
    permute_164: "f32[128, 128]" = torch.ops.aten.permute.default(arg637_1, [1, 0]);  arg637_1 = None
    addmm_124: "f32[128, 128]" = torch.ops.aten.addmm.default(arg638_1, view_328, permute_164);  arg638_1 = view_328 = permute_164 = None
    view_329: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_124, [1, 128, 128]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_333: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_329, [1, 128, 4, 32]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
    
    # No stacktrace found for following nodes
    clone_default_46: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_330: "f32[128, 512]" = torch.ops.aten.reshape.default(add_122, [128, 512])
    permute_165: "f32[512, 128]" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
    addmm_125: "f32[128, 128]" = torch.ops.aten.addmm.default(arg640_1, view_330, permute_165);  arg640_1 = view_330 = permute_165 = None
    view_331: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_125, [1, 128, 128]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_334: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_331, [1, 128, 4, 32]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_168: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # No stacktrace found for following nodes
    clone_default_47: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    _scaled_dot_product_flash_attention_default_15 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_45, clone_default_46, clone_default_47, scale = 0.17677669529663687);  clone_default_45 = clone_default_46 = clone_default_47 = None
    getitem_17: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_15[0];  _scaled_dot_product_flash_attention_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_170: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_17, [0, 2, 1, 3]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_341: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_170, [1, 128, 128]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_342: "f32[128, 128]" = torch.ops.aten.reshape.default(view_341, [128, 128]);  view_341 = None
    permute_171: "f32[128, 128]" = torch.ops.aten.permute.default(arg641_1, [1, 0]);  arg641_1 = None
    addmm_126: "f32[128, 128]" = torch.ops.aten.addmm.default(arg642_1, view_342, permute_171);  arg642_1 = view_342 = permute_171 = None
    view_343: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_126, [1, 128, 128]);  addmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_322: "f32[128, 512]" = torch.ops.aten.reshape.default(add_122, [128, 512])
    permute_161: "f32[512, 128]" = torch.ops.aten.permute.default(arg631_1, [1, 0]);  arg631_1 = None
    addmm_121: "f32[128, 128]" = torch.ops.aten.addmm.default(arg632_1, view_322, permute_161);  arg632_1 = view_322 = permute_161 = None
    view_323: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_121, [1, 128, 128]);  addmm_121 = None
    
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
    permute_172: "f32[128, 512]" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
    addmm_127: "f32[128, 512]" = torch.ops.aten.addmm.default(arg644_1, view_344, permute_172);  arg644_1 = view_344 = permute_172 = None
    view_345: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_127, [1, 128, 512]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_32: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_345);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_346: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_32, [128, 512]);  relu_32 = None
    permute_173: "f32[512, 128]" = torch.ops.aten.permute.default(arg645_1, [1, 0]);  arg645_1 = None
    addmm_128: "f32[128, 128]" = torch.ops.aten.addmm.default(arg646_1, view_346, permute_173);  arg646_1 = view_346 = permute_173 = None
    view_347: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_128, [1, 128, 128]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_128: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_347, add_127);  view_347 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_69: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_128, arg136_1);  add_128 = arg136_1 = None
    add_129: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_69, arg137_1);  mul_69 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_348: "f32[128, 128]" = torch.ops.aten.reshape.default(add_129, [128, 128])
    permute_174: "f32[128, 512]" = torch.ops.aten.permute.default(arg647_1, [1, 0]);  arg647_1 = None
    addmm_129: "f32[128, 512]" = torch.ops.aten.addmm.default(arg648_1, view_348, permute_174);  arg648_1 = view_348 = permute_174 = None
    view_349: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_129, [1, 128, 512]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_33: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_349);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_350: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_33, [128, 512]);  relu_33 = None
    permute_175: "f32[512, 128]" = torch.ops.aten.permute.default(arg649_1, [1, 0]);  arg649_1 = None
    addmm_130: "f32[128, 128]" = torch.ops.aten.addmm.default(arg650_1, view_350, permute_175);  arg650_1 = view_350 = permute_175 = None
    view_351: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_130, [1, 128, 128]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_130: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_351, add_129);  view_351 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_70: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_130, arg138_1);  add_130 = arg138_1 = None
    add_131: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_70, arg139_1);  mul_70 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_352: "f32[128, 128]" = torch.ops.aten.reshape.default(add_131, [128, 128])
    permute_176: "f32[128, 512]" = torch.ops.aten.permute.default(arg651_1, [1, 0]);  arg651_1 = None
    addmm_131: "f32[128, 512]" = torch.ops.aten.addmm.default(arg652_1, view_352, permute_176);  arg652_1 = view_352 = permute_176 = None
    view_353: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_131, [1, 128, 512]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_34: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_353);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_354: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_34, [128, 512]);  relu_34 = None
    permute_177: "f32[512, 128]" = torch.ops.aten.permute.default(arg653_1, [1, 0]);  arg653_1 = None
    addmm_132: "f32[128, 128]" = torch.ops.aten.addmm.default(arg654_1, view_354, permute_177);  arg654_1 = view_354 = permute_177 = None
    view_355: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_132, [1, 128, 128]);  addmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_132: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_355, add_131);  view_355 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_71: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_132, arg140_1);  add_132 = arg140_1 = None
    add_133: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_71, arg141_1);  mul_71 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[128, 128]" = torch.ops.aten.reshape.default(add_133, [128, 128])
    permute_178: "f32[128, 512]" = torch.ops.aten.permute.default(arg655_1, [1, 0]);  arg655_1 = None
    addmm_133: "f32[128, 512]" = torch.ops.aten.addmm.default(arg656_1, view_356, permute_178);  arg656_1 = view_356 = permute_178 = None
    view_357: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_133, [1, 128, 512]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_35: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_357);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_358: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_35, [128, 512]);  relu_35 = None
    permute_179: "f32[512, 128]" = torch.ops.aten.permute.default(arg657_1, [1, 0]);  arg657_1 = None
    addmm_134: "f32[128, 128]" = torch.ops.aten.addmm.default(arg658_1, view_358, permute_179);  arg658_1 = view_358 = permute_179 = None
    view_359: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_134, [1, 128, 128]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_134: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_133);  view_359 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_72: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_134, arg142_1);  add_134 = arg142_1 = None
    add_135: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_72, arg143_1);  mul_72 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_360: "f32[128, 128]" = torch.ops.aten.reshape.default(add_135, [128, 128]);  add_135 = None
    permute_180: "f32[128, 512]" = torch.ops.aten.permute.default(arg659_1, [1, 0]);  arg659_1 = None
    addmm_135: "f32[128, 512]" = torch.ops.aten.addmm.default(arg660_1, view_360, permute_180);  arg660_1 = view_360 = permute_180 = None
    view_361: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_135, [1, 128, 512]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_136: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_361, add_122);  view_361 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_73: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_136, arg144_1);  add_136 = arg144_1 = None
    add_137: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_73, arg145_1);  mul_73 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_364: "f32[128, 512]" = torch.ops.aten.reshape.default(add_137, [128, 512])
    permute_182: "f32[512, 128]" = torch.ops.aten.permute.default(arg663_1, [1, 0]);  arg663_1 = None
    addmm_137: "f32[128, 128]" = torch.ops.aten.addmm.default(arg664_1, view_364, permute_182);  arg664_1 = view_364 = permute_182 = None
    view_365: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_137, [1, 128, 128]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_75: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_365, arg148_1);  view_365 = arg148_1 = None
    add_139: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_75, arg149_1);  mul_75 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_366: "f32[128, 128]" = torch.ops.aten.reshape.default(add_139, [128, 128])
    permute_183: "f32[128, 128]" = torch.ops.aten.permute.default(arg665_1, [1, 0]);  arg665_1 = None
    addmm_138: "f32[128, 128]" = torch.ops.aten.addmm.default(arg666_1, view_366, permute_183);  arg666_1 = view_366 = permute_183 = None
    view_367: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_138, [1, 128, 128]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_372: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_367, [1, 128, 4, 32]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_186: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # No stacktrace found for following nodes
    clone_default_42: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_368: "f32[128, 128]" = torch.ops.aten.reshape.default(add_139, [128, 128]);  add_139 = None
    permute_184: "f32[128, 128]" = torch.ops.aten.permute.default(arg667_1, [1, 0]);  arg667_1 = None
    addmm_139: "f32[128, 128]" = torch.ops.aten.addmm.default(arg668_1, view_368, permute_184);  arg668_1 = view_368 = permute_184 = None
    view_369: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_139, [1, 128, 128]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_373: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_369, [1, 128, 4, 32]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_187: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # No stacktrace found for following nodes
    clone_default_43: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_370: "f32[128, 512]" = torch.ops.aten.reshape.default(add_137, [128, 512])
    permute_185: "f32[512, 128]" = torch.ops.aten.permute.default(arg669_1, [1, 0]);  arg669_1 = None
    addmm_140: "f32[128, 128]" = torch.ops.aten.addmm.default(arg670_1, view_370, permute_185);  arg670_1 = view_370 = permute_185 = None
    view_371: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_140, [1, 128, 128]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_374: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_371, [1, 128, 4, 32]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_188: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # No stacktrace found for following nodes
    clone_default_44: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    _scaled_dot_product_flash_attention_default_14 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_42, clone_default_43, clone_default_44, scale = 0.17677669529663687);  clone_default_42 = clone_default_43 = clone_default_44 = None
    getitem_16: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_14[0];  _scaled_dot_product_flash_attention_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_190: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_381: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_190, [1, 128, 128]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_382: "f32[128, 128]" = torch.ops.aten.reshape.default(view_381, [128, 128]);  view_381 = None
    permute_191: "f32[128, 128]" = torch.ops.aten.permute.default(arg671_1, [1, 0]);  arg671_1 = None
    addmm_141: "f32[128, 128]" = torch.ops.aten.addmm.default(arg672_1, view_382, permute_191);  arg672_1 = view_382 = permute_191 = None
    view_383: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_141, [1, 128, 128]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_362: "f32[128, 512]" = torch.ops.aten.reshape.default(add_137, [128, 512])
    permute_181: "f32[512, 128]" = torch.ops.aten.permute.default(arg661_1, [1, 0]);  arg661_1 = None
    addmm_136: "f32[128, 128]" = torch.ops.aten.addmm.default(arg662_1, view_362, permute_181);  arg662_1 = view_362 = permute_181 = None
    view_363: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_136, [1, 128, 128]);  addmm_136 = None
    
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
    permute_192: "f32[128, 512]" = torch.ops.aten.permute.default(arg673_1, [1, 0]);  arg673_1 = None
    addmm_142: "f32[128, 512]" = torch.ops.aten.addmm.default(arg674_1, view_384, permute_192);  arg674_1 = view_384 = permute_192 = None
    view_385: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_142, [1, 128, 512]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_36: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_385);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_386: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_36, [128, 512]);  relu_36 = None
    permute_193: "f32[512, 128]" = torch.ops.aten.permute.default(arg675_1, [1, 0]);  arg675_1 = None
    addmm_143: "f32[128, 128]" = torch.ops.aten.addmm.default(arg676_1, view_386, permute_193);  arg676_1 = view_386 = permute_193 = None
    view_387: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_143, [1, 128, 128]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_143: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_387, add_142);  view_387 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_77: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_143, arg152_1);  add_143 = arg152_1 = None
    add_144: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_77, arg153_1);  mul_77 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_388: "f32[128, 128]" = torch.ops.aten.reshape.default(add_144, [128, 128])
    permute_194: "f32[128, 512]" = torch.ops.aten.permute.default(arg677_1, [1, 0]);  arg677_1 = None
    addmm_144: "f32[128, 512]" = torch.ops.aten.addmm.default(arg678_1, view_388, permute_194);  arg678_1 = view_388 = permute_194 = None
    view_389: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_144, [1, 128, 512]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_37: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_389);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_390: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_37, [128, 512]);  relu_37 = None
    permute_195: "f32[512, 128]" = torch.ops.aten.permute.default(arg679_1, [1, 0]);  arg679_1 = None
    addmm_145: "f32[128, 128]" = torch.ops.aten.addmm.default(arg680_1, view_390, permute_195);  arg680_1 = view_390 = permute_195 = None
    view_391: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_145, [1, 128, 128]);  addmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_145: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_391, add_144);  view_391 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_78: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_145, arg154_1);  add_145 = arg154_1 = None
    add_146: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_78, arg155_1);  mul_78 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[128, 128]" = torch.ops.aten.reshape.default(add_146, [128, 128])
    permute_196: "f32[128, 512]" = torch.ops.aten.permute.default(arg681_1, [1, 0]);  arg681_1 = None
    addmm_146: "f32[128, 512]" = torch.ops.aten.addmm.default(arg682_1, view_392, permute_196);  arg682_1 = view_392 = permute_196 = None
    view_393: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_146, [1, 128, 512]);  addmm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_38: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_393);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_394: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_38, [128, 512]);  relu_38 = None
    permute_197: "f32[512, 128]" = torch.ops.aten.permute.default(arg683_1, [1, 0]);  arg683_1 = None
    addmm_147: "f32[128, 128]" = torch.ops.aten.addmm.default(arg684_1, view_394, permute_197);  arg684_1 = view_394 = permute_197 = None
    view_395: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_147, [1, 128, 128]);  addmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_147: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_395, add_146);  view_395 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_79: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_147, arg156_1);  add_147 = arg156_1 = None
    add_148: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, arg157_1);  mul_79 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_396: "f32[128, 128]" = torch.ops.aten.reshape.default(add_148, [128, 128])
    permute_198: "f32[128, 512]" = torch.ops.aten.permute.default(arg685_1, [1, 0]);  arg685_1 = None
    addmm_148: "f32[128, 512]" = torch.ops.aten.addmm.default(arg686_1, view_396, permute_198);  arg686_1 = view_396 = permute_198 = None
    view_397: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_148, [1, 128, 512]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_39: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_397);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_398: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_39, [128, 512]);  relu_39 = None
    permute_199: "f32[512, 128]" = torch.ops.aten.permute.default(arg687_1, [1, 0]);  arg687_1 = None
    addmm_149: "f32[128, 128]" = torch.ops.aten.addmm.default(arg688_1, view_398, permute_199);  arg688_1 = view_398 = permute_199 = None
    view_399: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_149, [1, 128, 128]);  addmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_149: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_399, add_148);  view_399 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_80: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_149, arg158_1);  add_149 = arg158_1 = None
    add_150: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_80, arg159_1);  mul_80 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_400: "f32[128, 128]" = torch.ops.aten.reshape.default(add_150, [128, 128]);  add_150 = None
    permute_200: "f32[128, 512]" = torch.ops.aten.permute.default(arg689_1, [1, 0]);  arg689_1 = None
    addmm_150: "f32[128, 512]" = torch.ops.aten.addmm.default(arg690_1, view_400, permute_200);  arg690_1 = view_400 = permute_200 = None
    view_401: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_150, [1, 128, 512]);  addmm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_151: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_401, add_137);  view_401 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_81: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_151, arg160_1);  add_151 = arg160_1 = None
    add_152: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_81, arg161_1);  mul_81 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_404: "f32[128, 512]" = torch.ops.aten.reshape.default(add_152, [128, 512])
    permute_202: "f32[512, 128]" = torch.ops.aten.permute.default(arg693_1, [1, 0]);  arg693_1 = None
    addmm_152: "f32[128, 128]" = torch.ops.aten.addmm.default(arg694_1, view_404, permute_202);  arg694_1 = view_404 = permute_202 = None
    view_405: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_152, [1, 128, 128]);  addmm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_83: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_405, arg164_1);  view_405 = arg164_1 = None
    add_154: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_83, arg165_1);  mul_83 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_406: "f32[128, 128]" = torch.ops.aten.reshape.default(add_154, [128, 128])
    permute_203: "f32[128, 128]" = torch.ops.aten.permute.default(arg695_1, [1, 0]);  arg695_1 = None
    addmm_153: "f32[128, 128]" = torch.ops.aten.addmm.default(arg696_1, view_406, permute_203);  arg696_1 = view_406 = permute_203 = None
    view_407: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_153, [1, 128, 128]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_412: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_407, [1, 128, 4, 32]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_206: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    
    # No stacktrace found for following nodes
    clone_default_39: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_408: "f32[128, 128]" = torch.ops.aten.reshape.default(add_154, [128, 128]);  add_154 = None
    permute_204: "f32[128, 128]" = torch.ops.aten.permute.default(arg697_1, [1, 0]);  arg697_1 = None
    addmm_154: "f32[128, 128]" = torch.ops.aten.addmm.default(arg698_1, view_408, permute_204);  arg698_1 = view_408 = permute_204 = None
    view_409: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_154, [1, 128, 128]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_413: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_409, [1, 128, 4, 32]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_207: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    
    # No stacktrace found for following nodes
    clone_default_40: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_410: "f32[128, 512]" = torch.ops.aten.reshape.default(add_152, [128, 512])
    permute_205: "f32[512, 128]" = torch.ops.aten.permute.default(arg699_1, [1, 0]);  arg699_1 = None
    addmm_155: "f32[128, 128]" = torch.ops.aten.addmm.default(arg700_1, view_410, permute_205);  arg700_1 = view_410 = permute_205 = None
    view_411: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_155, [1, 128, 128]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_414: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_411, [1, 128, 4, 32]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_208: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
    
    # No stacktrace found for following nodes
    clone_default_41: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    _scaled_dot_product_flash_attention_default_13 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_39, clone_default_40, clone_default_41, scale = 0.17677669529663687);  clone_default_39 = clone_default_40 = clone_default_41 = None
    getitem_15: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_13[0];  _scaled_dot_product_flash_attention_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_210: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_421: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_210, [1, 128, 128]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_422: "f32[128, 128]" = torch.ops.aten.reshape.default(view_421, [128, 128]);  view_421 = None
    permute_211: "f32[128, 128]" = torch.ops.aten.permute.default(arg701_1, [1, 0]);  arg701_1 = None
    addmm_156: "f32[128, 128]" = torch.ops.aten.addmm.default(arg702_1, view_422, permute_211);  arg702_1 = view_422 = permute_211 = None
    view_423: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_156, [1, 128, 128]);  addmm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_402: "f32[128, 512]" = torch.ops.aten.reshape.default(add_152, [128, 512])
    permute_201: "f32[512, 128]" = torch.ops.aten.permute.default(arg691_1, [1, 0]);  arg691_1 = None
    addmm_151: "f32[128, 128]" = torch.ops.aten.addmm.default(arg692_1, view_402, permute_201);  arg692_1 = view_402 = permute_201 = None
    view_403: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_151, [1, 128, 128]);  addmm_151 = None
    
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
    permute_212: "f32[128, 512]" = torch.ops.aten.permute.default(arg703_1, [1, 0]);  arg703_1 = None
    addmm_157: "f32[128, 512]" = torch.ops.aten.addmm.default(arg704_1, view_424, permute_212);  arg704_1 = view_424 = permute_212 = None
    view_425: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_157, [1, 128, 512]);  addmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_40: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_425);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_426: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_40, [128, 512]);  relu_40 = None
    permute_213: "f32[512, 128]" = torch.ops.aten.permute.default(arg705_1, [1, 0]);  arg705_1 = None
    addmm_158: "f32[128, 128]" = torch.ops.aten.addmm.default(arg706_1, view_426, permute_213);  arg706_1 = view_426 = permute_213 = None
    view_427: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_158, [1, 128, 128]);  addmm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_158: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_427, add_157);  view_427 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_85: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_158, arg168_1);  add_158 = arg168_1 = None
    add_159: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_85, arg169_1);  mul_85 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_428: "f32[128, 128]" = torch.ops.aten.reshape.default(add_159, [128, 128])
    permute_214: "f32[128, 512]" = torch.ops.aten.permute.default(arg707_1, [1, 0]);  arg707_1 = None
    addmm_159: "f32[128, 512]" = torch.ops.aten.addmm.default(arg708_1, view_428, permute_214);  arg708_1 = view_428 = permute_214 = None
    view_429: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_159, [1, 128, 512]);  addmm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_41: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_429);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_430: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_41, [128, 512]);  relu_41 = None
    permute_215: "f32[512, 128]" = torch.ops.aten.permute.default(arg709_1, [1, 0]);  arg709_1 = None
    addmm_160: "f32[128, 128]" = torch.ops.aten.addmm.default(arg710_1, view_430, permute_215);  arg710_1 = view_430 = permute_215 = None
    view_431: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_160, [1, 128, 128]);  addmm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_160: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_431, add_159);  view_431 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_86: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_160, arg170_1);  add_160 = arg170_1 = None
    add_161: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, arg171_1);  mul_86 = arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_432: "f32[128, 128]" = torch.ops.aten.reshape.default(add_161, [128, 128])
    permute_216: "f32[128, 512]" = torch.ops.aten.permute.default(arg711_1, [1, 0]);  arg711_1 = None
    addmm_161: "f32[128, 512]" = torch.ops.aten.addmm.default(arg712_1, view_432, permute_216);  arg712_1 = view_432 = permute_216 = None
    view_433: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_161, [1, 128, 512]);  addmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_42: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_433);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_434: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_42, [128, 512]);  relu_42 = None
    permute_217: "f32[512, 128]" = torch.ops.aten.permute.default(arg713_1, [1, 0]);  arg713_1 = None
    addmm_162: "f32[128, 128]" = torch.ops.aten.addmm.default(arg714_1, view_434, permute_217);  arg714_1 = view_434 = permute_217 = None
    view_435: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_162, [1, 128, 128]);  addmm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_162: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_435, add_161);  view_435 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_87: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_162, arg172_1);  add_162 = arg172_1 = None
    add_163: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_87, arg173_1);  mul_87 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_436: "f32[128, 128]" = torch.ops.aten.reshape.default(add_163, [128, 128])
    permute_218: "f32[128, 512]" = torch.ops.aten.permute.default(arg715_1, [1, 0]);  arg715_1 = None
    addmm_163: "f32[128, 512]" = torch.ops.aten.addmm.default(arg716_1, view_436, permute_218);  arg716_1 = view_436 = permute_218 = None
    view_437: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_163, [1, 128, 512]);  addmm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_43: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_437);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_438: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_43, [128, 512]);  relu_43 = None
    permute_219: "f32[512, 128]" = torch.ops.aten.permute.default(arg717_1, [1, 0]);  arg717_1 = None
    addmm_164: "f32[128, 128]" = torch.ops.aten.addmm.default(arg718_1, view_438, permute_219);  arg718_1 = view_438 = permute_219 = None
    view_439: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_164, [1, 128, 128]);  addmm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_164: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_439, add_163);  view_439 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_88: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_164, arg174_1);  add_164 = arg174_1 = None
    add_165: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_88, arg175_1);  mul_88 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_440: "f32[128, 128]" = torch.ops.aten.reshape.default(add_165, [128, 128]);  add_165 = None
    permute_220: "f32[128, 512]" = torch.ops.aten.permute.default(arg719_1, [1, 0]);  arg719_1 = None
    addmm_165: "f32[128, 512]" = torch.ops.aten.addmm.default(arg720_1, view_440, permute_220);  arg720_1 = view_440 = permute_220 = None
    view_441: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_165, [1, 128, 512]);  addmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_166: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_441, add_152);  view_441 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_89: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_166, arg176_1);  add_166 = arg176_1 = None
    add_167: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_89, arg177_1);  mul_89 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_444: "f32[128, 512]" = torch.ops.aten.reshape.default(add_167, [128, 512])
    permute_222: "f32[512, 128]" = torch.ops.aten.permute.default(arg723_1, [1, 0]);  arg723_1 = None
    addmm_167: "f32[128, 128]" = torch.ops.aten.addmm.default(arg724_1, view_444, permute_222);  arg724_1 = view_444 = permute_222 = None
    view_445: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_167, [1, 128, 128]);  addmm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_91: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_445, arg180_1);  view_445 = arg180_1 = None
    add_169: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_91, arg181_1);  mul_91 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_446: "f32[128, 128]" = torch.ops.aten.reshape.default(add_169, [128, 128])
    permute_223: "f32[128, 128]" = torch.ops.aten.permute.default(arg725_1, [1, 0]);  arg725_1 = None
    addmm_168: "f32[128, 128]" = torch.ops.aten.addmm.default(arg726_1, view_446, permute_223);  arg726_1 = view_446 = permute_223 = None
    view_447: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_168, [1, 128, 128]);  addmm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_452: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_447, [1, 128, 4, 32]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_226: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
    
    # No stacktrace found for following nodes
    clone_default_36: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_448: "f32[128, 128]" = torch.ops.aten.reshape.default(add_169, [128, 128]);  add_169 = None
    permute_224: "f32[128, 128]" = torch.ops.aten.permute.default(arg727_1, [1, 0]);  arg727_1 = None
    addmm_169: "f32[128, 128]" = torch.ops.aten.addmm.default(arg728_1, view_448, permute_224);  arg728_1 = view_448 = permute_224 = None
    view_449: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_169, [1, 128, 128]);  addmm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_453: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_449, [1, 128, 4, 32]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # No stacktrace found for following nodes
    clone_default_37: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_450: "f32[128, 512]" = torch.ops.aten.reshape.default(add_167, [128, 512])
    permute_225: "f32[512, 128]" = torch.ops.aten.permute.default(arg729_1, [1, 0]);  arg729_1 = None
    addmm_170: "f32[128, 128]" = torch.ops.aten.addmm.default(arg730_1, view_450, permute_225);  arg730_1 = view_450 = permute_225 = None
    view_451: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_170, [1, 128, 128]);  addmm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_454: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_451, [1, 128, 4, 32]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # No stacktrace found for following nodes
    clone_default_38: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    _scaled_dot_product_flash_attention_default_12 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_36, clone_default_37, clone_default_38, scale = 0.17677669529663687);  clone_default_36 = clone_default_37 = clone_default_38 = None
    getitem_14: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_12[0];  _scaled_dot_product_flash_attention_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_230: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3]);  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_461: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_230, [1, 128, 128]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_462: "f32[128, 128]" = torch.ops.aten.reshape.default(view_461, [128, 128]);  view_461 = None
    permute_231: "f32[128, 128]" = torch.ops.aten.permute.default(arg731_1, [1, 0]);  arg731_1 = None
    addmm_171: "f32[128, 128]" = torch.ops.aten.addmm.default(arg732_1, view_462, permute_231);  arg732_1 = view_462 = permute_231 = None
    view_463: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_171, [1, 128, 128]);  addmm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_442: "f32[128, 512]" = torch.ops.aten.reshape.default(add_167, [128, 512])
    permute_221: "f32[512, 128]" = torch.ops.aten.permute.default(arg721_1, [1, 0]);  arg721_1 = None
    addmm_166: "f32[128, 128]" = torch.ops.aten.addmm.default(arg722_1, view_442, permute_221);  arg722_1 = view_442 = permute_221 = None
    view_443: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_166, [1, 128, 128]);  addmm_166 = None
    
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
    permute_232: "f32[128, 512]" = torch.ops.aten.permute.default(arg733_1, [1, 0]);  arg733_1 = None
    addmm_172: "f32[128, 512]" = torch.ops.aten.addmm.default(arg734_1, view_464, permute_232);  arg734_1 = view_464 = permute_232 = None
    view_465: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_172, [1, 128, 512]);  addmm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_44: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_465);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_466: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_44, [128, 512]);  relu_44 = None
    permute_233: "f32[512, 128]" = torch.ops.aten.permute.default(arg735_1, [1, 0]);  arg735_1 = None
    addmm_173: "f32[128, 128]" = torch.ops.aten.addmm.default(arg736_1, view_466, permute_233);  arg736_1 = view_466 = permute_233 = None
    view_467: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_173, [1, 128, 128]);  addmm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_173: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_467, add_172);  view_467 = add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_93: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_173, arg184_1);  add_173 = arg184_1 = None
    add_174: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_93, arg185_1);  mul_93 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_468: "f32[128, 128]" = torch.ops.aten.reshape.default(add_174, [128, 128])
    permute_234: "f32[128, 512]" = torch.ops.aten.permute.default(arg737_1, [1, 0]);  arg737_1 = None
    addmm_174: "f32[128, 512]" = torch.ops.aten.addmm.default(arg738_1, view_468, permute_234);  arg738_1 = view_468 = permute_234 = None
    view_469: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_174, [1, 128, 512]);  addmm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_45: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_469);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_470: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_45, [128, 512]);  relu_45 = None
    permute_235: "f32[512, 128]" = torch.ops.aten.permute.default(arg739_1, [1, 0]);  arg739_1 = None
    addmm_175: "f32[128, 128]" = torch.ops.aten.addmm.default(arg740_1, view_470, permute_235);  arg740_1 = view_470 = permute_235 = None
    view_471: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_175, [1, 128, 128]);  addmm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_175: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_471, add_174);  view_471 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_94: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_175, arg186_1);  add_175 = arg186_1 = None
    add_176: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_94, arg187_1);  mul_94 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_472: "f32[128, 128]" = torch.ops.aten.reshape.default(add_176, [128, 128])
    permute_236: "f32[128, 512]" = torch.ops.aten.permute.default(arg741_1, [1, 0]);  arg741_1 = None
    addmm_176: "f32[128, 512]" = torch.ops.aten.addmm.default(arg742_1, view_472, permute_236);  arg742_1 = view_472 = permute_236 = None
    view_473: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_176, [1, 128, 512]);  addmm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_46: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_473);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_474: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_46, [128, 512]);  relu_46 = None
    permute_237: "f32[512, 128]" = torch.ops.aten.permute.default(arg743_1, [1, 0]);  arg743_1 = None
    addmm_177: "f32[128, 128]" = torch.ops.aten.addmm.default(arg744_1, view_474, permute_237);  arg744_1 = view_474 = permute_237 = None
    view_475: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_177, [1, 128, 128]);  addmm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_177: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_475, add_176);  view_475 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_95: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_177, arg188_1);  add_177 = arg188_1 = None
    add_178: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_95, arg189_1);  mul_95 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[128, 128]" = torch.ops.aten.reshape.default(add_178, [128, 128])
    permute_238: "f32[128, 512]" = torch.ops.aten.permute.default(arg745_1, [1, 0]);  arg745_1 = None
    addmm_178: "f32[128, 512]" = torch.ops.aten.addmm.default(arg746_1, view_476, permute_238);  arg746_1 = view_476 = permute_238 = None
    view_477: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_178, [1, 128, 512]);  addmm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_47: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_477);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_478: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_47, [128, 512]);  relu_47 = None
    permute_239: "f32[512, 128]" = torch.ops.aten.permute.default(arg747_1, [1, 0]);  arg747_1 = None
    addmm_179: "f32[128, 128]" = torch.ops.aten.addmm.default(arg748_1, view_478, permute_239);  arg748_1 = view_478 = permute_239 = None
    view_479: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_179, [1, 128, 128]);  addmm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_179: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_479, add_178);  view_479 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_96: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_179, arg190_1);  add_179 = arg190_1 = None
    add_180: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_96, arg191_1);  mul_96 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_480: "f32[128, 128]" = torch.ops.aten.reshape.default(add_180, [128, 128]);  add_180 = None
    permute_240: "f32[128, 512]" = torch.ops.aten.permute.default(arg749_1, [1, 0]);  arg749_1 = None
    addmm_180: "f32[128, 512]" = torch.ops.aten.addmm.default(arg750_1, view_480, permute_240);  arg750_1 = view_480 = permute_240 = None
    view_481: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_180, [1, 128, 512]);  addmm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_181: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_481, add_167);  view_481 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_97: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_181, arg192_1);  add_181 = arg192_1 = None
    add_182: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_97, arg193_1);  mul_97 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_484: "f32[128, 512]" = torch.ops.aten.reshape.default(add_182, [128, 512])
    permute_242: "f32[512, 128]" = torch.ops.aten.permute.default(arg753_1, [1, 0]);  arg753_1 = None
    addmm_182: "f32[128, 128]" = torch.ops.aten.addmm.default(arg754_1, view_484, permute_242);  arg754_1 = view_484 = permute_242 = None
    view_485: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_182, [1, 128, 128]);  addmm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_99: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_485, arg196_1);  view_485 = arg196_1 = None
    add_184: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_99, arg197_1);  mul_99 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_486: "f32[128, 128]" = torch.ops.aten.reshape.default(add_184, [128, 128])
    permute_243: "f32[128, 128]" = torch.ops.aten.permute.default(arg755_1, [1, 0]);  arg755_1 = None
    addmm_183: "f32[128, 128]" = torch.ops.aten.addmm.default(arg756_1, view_486, permute_243);  arg756_1 = view_486 = permute_243 = None
    view_487: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_183, [1, 128, 128]);  addmm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_492: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_487, [1, 128, 4, 32]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_246: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # No stacktrace found for following nodes
    clone_default_33: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_488: "f32[128, 128]" = torch.ops.aten.reshape.default(add_184, [128, 128]);  add_184 = None
    permute_244: "f32[128, 128]" = torch.ops.aten.permute.default(arg757_1, [1, 0]);  arg757_1 = None
    addmm_184: "f32[128, 128]" = torch.ops.aten.addmm.default(arg758_1, view_488, permute_244);  arg758_1 = view_488 = permute_244 = None
    view_489: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_184, [1, 128, 128]);  addmm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_493: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_489, [1, 128, 4, 32]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_247: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # No stacktrace found for following nodes
    clone_default_34: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_490: "f32[128, 512]" = torch.ops.aten.reshape.default(add_182, [128, 512])
    permute_245: "f32[512, 128]" = torch.ops.aten.permute.default(arg759_1, [1, 0]);  arg759_1 = None
    addmm_185: "f32[128, 128]" = torch.ops.aten.addmm.default(arg760_1, view_490, permute_245);  arg760_1 = view_490 = permute_245 = None
    view_491: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_185, [1, 128, 128]);  addmm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_494: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_491, [1, 128, 4, 32]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_248: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # No stacktrace found for following nodes
    clone_default_35: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_33, clone_default_34, clone_default_35, scale = 0.17677669529663687);  clone_default_33 = clone_default_34 = clone_default_35 = None
    getitem_13: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_11[0];  _scaled_dot_product_flash_attention_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_250: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3]);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_501: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_250, [1, 128, 128]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_502: "f32[128, 128]" = torch.ops.aten.reshape.default(view_501, [128, 128]);  view_501 = None
    permute_251: "f32[128, 128]" = torch.ops.aten.permute.default(arg761_1, [1, 0]);  arg761_1 = None
    addmm_186: "f32[128, 128]" = torch.ops.aten.addmm.default(arg762_1, view_502, permute_251);  arg762_1 = view_502 = permute_251 = None
    view_503: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_186, [1, 128, 128]);  addmm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_482: "f32[128, 512]" = torch.ops.aten.reshape.default(add_182, [128, 512])
    permute_241: "f32[512, 128]" = torch.ops.aten.permute.default(arg751_1, [1, 0]);  arg751_1 = None
    addmm_181: "f32[128, 128]" = torch.ops.aten.addmm.default(arg752_1, view_482, permute_241);  arg752_1 = view_482 = permute_241 = None
    view_483: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_181, [1, 128, 128]);  addmm_181 = None
    
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
    permute_252: "f32[128, 512]" = torch.ops.aten.permute.default(arg763_1, [1, 0]);  arg763_1 = None
    addmm_187: "f32[128, 512]" = torch.ops.aten.addmm.default(arg764_1, view_504, permute_252);  arg764_1 = view_504 = permute_252 = None
    view_505: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_187, [1, 128, 512]);  addmm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_48: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_505);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_506: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_48, [128, 512]);  relu_48 = None
    permute_253: "f32[512, 128]" = torch.ops.aten.permute.default(arg765_1, [1, 0]);  arg765_1 = None
    addmm_188: "f32[128, 128]" = torch.ops.aten.addmm.default(arg766_1, view_506, permute_253);  arg766_1 = view_506 = permute_253 = None
    view_507: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_188, [1, 128, 128]);  addmm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_188: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_507, add_187);  view_507 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_101: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_188, arg200_1);  add_188 = arg200_1 = None
    add_189: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, arg201_1);  mul_101 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_508: "f32[128, 128]" = torch.ops.aten.reshape.default(add_189, [128, 128])
    permute_254: "f32[128, 512]" = torch.ops.aten.permute.default(arg767_1, [1, 0]);  arg767_1 = None
    addmm_189: "f32[128, 512]" = torch.ops.aten.addmm.default(arg768_1, view_508, permute_254);  arg768_1 = view_508 = permute_254 = None
    view_509: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_189, [1, 128, 512]);  addmm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_49: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_509);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_510: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_49, [128, 512]);  relu_49 = None
    permute_255: "f32[512, 128]" = torch.ops.aten.permute.default(arg769_1, [1, 0]);  arg769_1 = None
    addmm_190: "f32[128, 128]" = torch.ops.aten.addmm.default(arg770_1, view_510, permute_255);  arg770_1 = view_510 = permute_255 = None
    view_511: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_190, [1, 128, 128]);  addmm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_190: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_511, add_189);  view_511 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_102: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_190, arg202_1);  add_190 = arg202_1 = None
    add_191: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_102, arg203_1);  mul_102 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_512: "f32[128, 128]" = torch.ops.aten.reshape.default(add_191, [128, 128])
    permute_256: "f32[128, 512]" = torch.ops.aten.permute.default(arg771_1, [1, 0]);  arg771_1 = None
    addmm_191: "f32[128, 512]" = torch.ops.aten.addmm.default(arg772_1, view_512, permute_256);  arg772_1 = view_512 = permute_256 = None
    view_513: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_191, [1, 128, 512]);  addmm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_50: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_513);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_514: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_50, [128, 512]);  relu_50 = None
    permute_257: "f32[512, 128]" = torch.ops.aten.permute.default(arg773_1, [1, 0]);  arg773_1 = None
    addmm_192: "f32[128, 128]" = torch.ops.aten.addmm.default(arg774_1, view_514, permute_257);  arg774_1 = view_514 = permute_257 = None
    view_515: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_192, [1, 128, 128]);  addmm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_192: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_515, add_191);  view_515 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_103: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_192, arg204_1);  add_192 = arg204_1 = None
    add_193: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_103, arg205_1);  mul_103 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_516: "f32[128, 128]" = torch.ops.aten.reshape.default(add_193, [128, 128])
    permute_258: "f32[128, 512]" = torch.ops.aten.permute.default(arg775_1, [1, 0]);  arg775_1 = None
    addmm_193: "f32[128, 512]" = torch.ops.aten.addmm.default(arg776_1, view_516, permute_258);  arg776_1 = view_516 = permute_258 = None
    view_517: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_193, [1, 128, 512]);  addmm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_51: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_517);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_518: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_51, [128, 512]);  relu_51 = None
    permute_259: "f32[512, 128]" = torch.ops.aten.permute.default(arg777_1, [1, 0]);  arg777_1 = None
    addmm_194: "f32[128, 128]" = torch.ops.aten.addmm.default(arg778_1, view_518, permute_259);  arg778_1 = view_518 = permute_259 = None
    view_519: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_194, [1, 128, 128]);  addmm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_194: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_519, add_193);  view_519 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_104: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_194, arg206_1);  add_194 = arg206_1 = None
    add_195: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_104, arg207_1);  mul_104 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_520: "f32[128, 128]" = torch.ops.aten.reshape.default(add_195, [128, 128]);  add_195 = None
    permute_260: "f32[128, 512]" = torch.ops.aten.permute.default(arg779_1, [1, 0]);  arg779_1 = None
    addmm_195: "f32[128, 512]" = torch.ops.aten.addmm.default(arg780_1, view_520, permute_260);  arg780_1 = view_520 = permute_260 = None
    view_521: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_195, [1, 128, 512]);  addmm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_196: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_521, add_182);  view_521 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_105: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_196, arg208_1);  add_196 = arg208_1 = None
    add_197: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_105, arg209_1);  mul_105 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_524: "f32[128, 512]" = torch.ops.aten.reshape.default(add_197, [128, 512])
    permute_262: "f32[512, 128]" = torch.ops.aten.permute.default(arg783_1, [1, 0]);  arg783_1 = None
    addmm_197: "f32[128, 128]" = torch.ops.aten.addmm.default(arg784_1, view_524, permute_262);  arg784_1 = view_524 = permute_262 = None
    view_525: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_197, [1, 128, 128]);  addmm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_107: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_525, arg212_1);  view_525 = arg212_1 = None
    add_199: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_107, arg213_1);  mul_107 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_526: "f32[128, 128]" = torch.ops.aten.reshape.default(add_199, [128, 128])
    permute_263: "f32[128, 128]" = torch.ops.aten.permute.default(arg785_1, [1, 0]);  arg785_1 = None
    addmm_198: "f32[128, 128]" = torch.ops.aten.addmm.default(arg786_1, view_526, permute_263);  arg786_1 = view_526 = permute_263 = None
    view_527: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_198, [1, 128, 128]);  addmm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_532: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_527, [1, 128, 4, 32]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_266: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    
    # No stacktrace found for following nodes
    clone_default_30: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_528: "f32[128, 128]" = torch.ops.aten.reshape.default(add_199, [128, 128]);  add_199 = None
    permute_264: "f32[128, 128]" = torch.ops.aten.permute.default(arg787_1, [1, 0]);  arg787_1 = None
    addmm_199: "f32[128, 128]" = torch.ops.aten.addmm.default(arg788_1, view_528, permute_264);  arg788_1 = view_528 = permute_264 = None
    view_529: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_199, [1, 128, 128]);  addmm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_533: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_529, [1, 128, 4, 32]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_267: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
    
    # No stacktrace found for following nodes
    clone_default_31: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_530: "f32[128, 512]" = torch.ops.aten.reshape.default(add_197, [128, 512])
    permute_265: "f32[512, 128]" = torch.ops.aten.permute.default(arg789_1, [1, 0]);  arg789_1 = None
    addmm_200: "f32[128, 128]" = torch.ops.aten.addmm.default(arg790_1, view_530, permute_265);  arg790_1 = view_530 = permute_265 = None
    view_531: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_200, [1, 128, 128]);  addmm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_534: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_531, [1, 128, 4, 32]);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_268: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    
    # No stacktrace found for following nodes
    clone_default_32: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_30, clone_default_31, clone_default_32, scale = 0.17677669529663687);  clone_default_30 = clone_default_31 = clone_default_32 = None
    getitem_12: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_10[0];  _scaled_dot_product_flash_attention_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_270: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_541: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_270, [1, 128, 128]);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_542: "f32[128, 128]" = torch.ops.aten.reshape.default(view_541, [128, 128]);  view_541 = None
    permute_271: "f32[128, 128]" = torch.ops.aten.permute.default(arg791_1, [1, 0]);  arg791_1 = None
    addmm_201: "f32[128, 128]" = torch.ops.aten.addmm.default(arg792_1, view_542, permute_271);  arg792_1 = view_542 = permute_271 = None
    view_543: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_201, [1, 128, 128]);  addmm_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_522: "f32[128, 512]" = torch.ops.aten.reshape.default(add_197, [128, 512])
    permute_261: "f32[512, 128]" = torch.ops.aten.permute.default(arg781_1, [1, 0]);  arg781_1 = None
    addmm_196: "f32[128, 128]" = torch.ops.aten.addmm.default(arg782_1, view_522, permute_261);  arg782_1 = view_522 = permute_261 = None
    view_523: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_196, [1, 128, 128]);  addmm_196 = None
    
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
    permute_272: "f32[128, 512]" = torch.ops.aten.permute.default(arg793_1, [1, 0]);  arg793_1 = None
    addmm_202: "f32[128, 512]" = torch.ops.aten.addmm.default(arg794_1, view_544, permute_272);  arg794_1 = view_544 = permute_272 = None
    view_545: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_202, [1, 128, 512]);  addmm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_52: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_545);  view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_546: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_52, [128, 512]);  relu_52 = None
    permute_273: "f32[512, 128]" = torch.ops.aten.permute.default(arg795_1, [1, 0]);  arg795_1 = None
    addmm_203: "f32[128, 128]" = torch.ops.aten.addmm.default(arg796_1, view_546, permute_273);  arg796_1 = view_546 = permute_273 = None
    view_547: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_203, [1, 128, 128]);  addmm_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_203: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_547, add_202);  view_547 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_109: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_203, arg216_1);  add_203 = arg216_1 = None
    add_204: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, arg217_1);  mul_109 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_548: "f32[128, 128]" = torch.ops.aten.reshape.default(add_204, [128, 128])
    permute_274: "f32[128, 512]" = torch.ops.aten.permute.default(arg797_1, [1, 0]);  arg797_1 = None
    addmm_204: "f32[128, 512]" = torch.ops.aten.addmm.default(arg798_1, view_548, permute_274);  arg798_1 = view_548 = permute_274 = None
    view_549: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_204, [1, 128, 512]);  addmm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_53: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_549);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_550: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_53, [128, 512]);  relu_53 = None
    permute_275: "f32[512, 128]" = torch.ops.aten.permute.default(arg799_1, [1, 0]);  arg799_1 = None
    addmm_205: "f32[128, 128]" = torch.ops.aten.addmm.default(arg800_1, view_550, permute_275);  arg800_1 = view_550 = permute_275 = None
    view_551: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_205, [1, 128, 128]);  addmm_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_205: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_551, add_204);  view_551 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_110: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_205, arg218_1);  add_205 = arg218_1 = None
    add_206: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_110, arg219_1);  mul_110 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_552: "f32[128, 128]" = torch.ops.aten.reshape.default(add_206, [128, 128])
    permute_276: "f32[128, 512]" = torch.ops.aten.permute.default(arg801_1, [1, 0]);  arg801_1 = None
    addmm_206: "f32[128, 512]" = torch.ops.aten.addmm.default(arg802_1, view_552, permute_276);  arg802_1 = view_552 = permute_276 = None
    view_553: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_206, [1, 128, 512]);  addmm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_54: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_553);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_554: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_54, [128, 512]);  relu_54 = None
    permute_277: "f32[512, 128]" = torch.ops.aten.permute.default(arg803_1, [1, 0]);  arg803_1 = None
    addmm_207: "f32[128, 128]" = torch.ops.aten.addmm.default(arg804_1, view_554, permute_277);  arg804_1 = view_554 = permute_277 = None
    view_555: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_207, [1, 128, 128]);  addmm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_207: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_555, add_206);  view_555 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_111: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_207, arg220_1);  add_207 = arg220_1 = None
    add_208: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_111, arg221_1);  mul_111 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_556: "f32[128, 128]" = torch.ops.aten.reshape.default(add_208, [128, 128])
    permute_278: "f32[128, 512]" = torch.ops.aten.permute.default(arg805_1, [1, 0]);  arg805_1 = None
    addmm_208: "f32[128, 512]" = torch.ops.aten.addmm.default(arg806_1, view_556, permute_278);  arg806_1 = view_556 = permute_278 = None
    view_557: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_208, [1, 128, 512]);  addmm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_55: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_557);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_558: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_55, [128, 512]);  relu_55 = None
    permute_279: "f32[512, 128]" = torch.ops.aten.permute.default(arg807_1, [1, 0]);  arg807_1 = None
    addmm_209: "f32[128, 128]" = torch.ops.aten.addmm.default(arg808_1, view_558, permute_279);  arg808_1 = view_558 = permute_279 = None
    view_559: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_209, [1, 128, 128]);  addmm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_209: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_559, add_208);  view_559 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_112: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_209, arg222_1);  add_209 = arg222_1 = None
    add_210: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_112, arg223_1);  mul_112 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_560: "f32[128, 128]" = torch.ops.aten.reshape.default(add_210, [128, 128]);  add_210 = None
    permute_280: "f32[128, 512]" = torch.ops.aten.permute.default(arg809_1, [1, 0]);  arg809_1 = None
    addmm_210: "f32[128, 512]" = torch.ops.aten.addmm.default(arg810_1, view_560, permute_280);  arg810_1 = view_560 = permute_280 = None
    view_561: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_210, [1, 128, 512]);  addmm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_211: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_561, add_197);  view_561 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_113: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_211, arg224_1);  add_211 = arg224_1 = None
    add_212: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_113, arg225_1);  mul_113 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_564: "f32[128, 512]" = torch.ops.aten.reshape.default(add_212, [128, 512])
    permute_282: "f32[512, 128]" = torch.ops.aten.permute.default(arg813_1, [1, 0]);  arg813_1 = None
    addmm_212: "f32[128, 128]" = torch.ops.aten.addmm.default(arg814_1, view_564, permute_282);  arg814_1 = view_564 = permute_282 = None
    view_565: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_212, [1, 128, 128]);  addmm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_115: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_565, arg228_1);  view_565 = arg228_1 = None
    add_214: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_115, arg229_1);  mul_115 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_566: "f32[128, 128]" = torch.ops.aten.reshape.default(add_214, [128, 128])
    permute_283: "f32[128, 128]" = torch.ops.aten.permute.default(arg815_1, [1, 0]);  arg815_1 = None
    addmm_213: "f32[128, 128]" = torch.ops.aten.addmm.default(arg816_1, view_566, permute_283);  arg816_1 = view_566 = permute_283 = None
    view_567: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_213, [1, 128, 128]);  addmm_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_572: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_567, [1, 128, 4, 32]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_286: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
    
    # No stacktrace found for following nodes
    clone_default_27: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_568: "f32[128, 128]" = torch.ops.aten.reshape.default(add_214, [128, 128]);  add_214 = None
    permute_284: "f32[128, 128]" = torch.ops.aten.permute.default(arg817_1, [1, 0]);  arg817_1 = None
    addmm_214: "f32[128, 128]" = torch.ops.aten.addmm.default(arg818_1, view_568, permute_284);  arg818_1 = view_568 = permute_284 = None
    view_569: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_214, [1, 128, 128]);  addmm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_573: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_569, [1, 128, 4, 32]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_287: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    
    # No stacktrace found for following nodes
    clone_default_28: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_570: "f32[128, 512]" = torch.ops.aten.reshape.default(add_212, [128, 512])
    permute_285: "f32[512, 128]" = torch.ops.aten.permute.default(arg819_1, [1, 0]);  arg819_1 = None
    addmm_215: "f32[128, 128]" = torch.ops.aten.addmm.default(arg820_1, view_570, permute_285);  arg820_1 = view_570 = permute_285 = None
    view_571: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_215, [1, 128, 128]);  addmm_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_574: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_571, [1, 128, 4, 32]);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    
    # No stacktrace found for following nodes
    clone_default_29: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_27, clone_default_28, clone_default_29, scale = 0.17677669529663687);  clone_default_27 = clone_default_28 = clone_default_29 = None
    getitem_11: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_9[0];  _scaled_dot_product_flash_attention_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_290: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_581: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_290, [1, 128, 128]);  permute_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_582: "f32[128, 128]" = torch.ops.aten.reshape.default(view_581, [128, 128]);  view_581 = None
    permute_291: "f32[128, 128]" = torch.ops.aten.permute.default(arg821_1, [1, 0]);  arg821_1 = None
    addmm_216: "f32[128, 128]" = torch.ops.aten.addmm.default(arg822_1, view_582, permute_291);  arg822_1 = view_582 = permute_291 = None
    view_583: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_216, [1, 128, 128]);  addmm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_562: "f32[128, 512]" = torch.ops.aten.reshape.default(add_212, [128, 512])
    permute_281: "f32[512, 128]" = torch.ops.aten.permute.default(arg811_1, [1, 0]);  arg811_1 = None
    addmm_211: "f32[128, 128]" = torch.ops.aten.addmm.default(arg812_1, view_562, permute_281);  arg812_1 = view_562 = permute_281 = None
    view_563: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_211, [1, 128, 128]);  addmm_211 = None
    
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
    permute_292: "f32[128, 512]" = torch.ops.aten.permute.default(arg823_1, [1, 0]);  arg823_1 = None
    addmm_217: "f32[128, 512]" = torch.ops.aten.addmm.default(arg824_1, view_584, permute_292);  arg824_1 = view_584 = permute_292 = None
    view_585: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_217, [1, 128, 512]);  addmm_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_56: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_585);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_586: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_56, [128, 512]);  relu_56 = None
    permute_293: "f32[512, 128]" = torch.ops.aten.permute.default(arg825_1, [1, 0]);  arg825_1 = None
    addmm_218: "f32[128, 128]" = torch.ops.aten.addmm.default(arg826_1, view_586, permute_293);  arg826_1 = view_586 = permute_293 = None
    view_587: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_218, [1, 128, 128]);  addmm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_218: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_587, add_217);  view_587 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_117: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_218, arg232_1);  add_218 = arg232_1 = None
    add_219: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_117, arg233_1);  mul_117 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_588: "f32[128, 128]" = torch.ops.aten.reshape.default(add_219, [128, 128])
    permute_294: "f32[128, 512]" = torch.ops.aten.permute.default(arg827_1, [1, 0]);  arg827_1 = None
    addmm_219: "f32[128, 512]" = torch.ops.aten.addmm.default(arg828_1, view_588, permute_294);  arg828_1 = view_588 = permute_294 = None
    view_589: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_219, [1, 128, 512]);  addmm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_57: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_589);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_590: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_57, [128, 512]);  relu_57 = None
    permute_295: "f32[512, 128]" = torch.ops.aten.permute.default(arg829_1, [1, 0]);  arg829_1 = None
    addmm_220: "f32[128, 128]" = torch.ops.aten.addmm.default(arg830_1, view_590, permute_295);  arg830_1 = view_590 = permute_295 = None
    view_591: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_220, [1, 128, 128]);  addmm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_220: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_591, add_219);  view_591 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_118: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_220, arg234_1);  add_220 = arg234_1 = None
    add_221: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_118, arg235_1);  mul_118 = arg235_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_592: "f32[128, 128]" = torch.ops.aten.reshape.default(add_221, [128, 128])
    permute_296: "f32[128, 512]" = torch.ops.aten.permute.default(arg831_1, [1, 0]);  arg831_1 = None
    addmm_221: "f32[128, 512]" = torch.ops.aten.addmm.default(arg832_1, view_592, permute_296);  arg832_1 = view_592 = permute_296 = None
    view_593: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_221, [1, 128, 512]);  addmm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_58: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_593);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_594: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_58, [128, 512]);  relu_58 = None
    permute_297: "f32[512, 128]" = torch.ops.aten.permute.default(arg833_1, [1, 0]);  arg833_1 = None
    addmm_222: "f32[128, 128]" = torch.ops.aten.addmm.default(arg834_1, view_594, permute_297);  arg834_1 = view_594 = permute_297 = None
    view_595: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_222, [1, 128, 128]);  addmm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_222: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_595, add_221);  view_595 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_119: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_222, arg236_1);  add_222 = arg236_1 = None
    add_223: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, arg237_1);  mul_119 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_596: "f32[128, 128]" = torch.ops.aten.reshape.default(add_223, [128, 128])
    permute_298: "f32[128, 512]" = torch.ops.aten.permute.default(arg835_1, [1, 0]);  arg835_1 = None
    addmm_223: "f32[128, 512]" = torch.ops.aten.addmm.default(arg836_1, view_596, permute_298);  arg836_1 = view_596 = permute_298 = None
    view_597: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_223, [1, 128, 512]);  addmm_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_59: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_597);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_598: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_59, [128, 512]);  relu_59 = None
    permute_299: "f32[512, 128]" = torch.ops.aten.permute.default(arg837_1, [1, 0]);  arg837_1 = None
    addmm_224: "f32[128, 128]" = torch.ops.aten.addmm.default(arg838_1, view_598, permute_299);  arg838_1 = view_598 = permute_299 = None
    view_599: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_224, [1, 128, 128]);  addmm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_224: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_599, add_223);  view_599 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_120: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_224, arg238_1);  add_224 = arg238_1 = None
    add_225: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_120, arg239_1);  mul_120 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_600: "f32[128, 128]" = torch.ops.aten.reshape.default(add_225, [128, 128]);  add_225 = None
    permute_300: "f32[128, 512]" = torch.ops.aten.permute.default(arg839_1, [1, 0]);  arg839_1 = None
    addmm_225: "f32[128, 512]" = torch.ops.aten.addmm.default(arg840_1, view_600, permute_300);  arg840_1 = view_600 = permute_300 = None
    view_601: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_225, [1, 128, 512]);  addmm_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_226: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_601, add_212);  view_601 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_121: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_226, arg240_1);  add_226 = arg240_1 = None
    add_227: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_121, arg241_1);  mul_121 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_604: "f32[128, 512]" = torch.ops.aten.reshape.default(add_227, [128, 512])
    permute_302: "f32[512, 128]" = torch.ops.aten.permute.default(arg843_1, [1, 0]);  arg843_1 = None
    addmm_227: "f32[128, 128]" = torch.ops.aten.addmm.default(arg844_1, view_604, permute_302);  arg844_1 = view_604 = permute_302 = None
    view_605: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_227, [1, 128, 128]);  addmm_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_123: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_605, arg244_1);  view_605 = arg244_1 = None
    add_229: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_123, arg245_1);  mul_123 = arg245_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_606: "f32[128, 128]" = torch.ops.aten.reshape.default(add_229, [128, 128])
    permute_303: "f32[128, 128]" = torch.ops.aten.permute.default(arg845_1, [1, 0]);  arg845_1 = None
    addmm_228: "f32[128, 128]" = torch.ops.aten.addmm.default(arg846_1, view_606, permute_303);  arg846_1 = view_606 = permute_303 = None
    view_607: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_228, [1, 128, 128]);  addmm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_612: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_607, [1, 128, 4, 32]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_306: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_608: "f32[128, 128]" = torch.ops.aten.reshape.default(add_229, [128, 128]);  add_229 = None
    permute_304: "f32[128, 128]" = torch.ops.aten.permute.default(arg847_1, [1, 0]);  arg847_1 = None
    addmm_229: "f32[128, 128]" = torch.ops.aten.addmm.default(arg848_1, view_608, permute_304);  arg848_1 = view_608 = permute_304 = None
    view_609: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_229, [1, 128, 128]);  addmm_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_613: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_609, [1, 128, 4, 32]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_307: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
    
    # No stacktrace found for following nodes
    clone_default_25: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_610: "f32[128, 512]" = torch.ops.aten.reshape.default(add_227, [128, 512])
    permute_305: "f32[512, 128]" = torch.ops.aten.permute.default(arg849_1, [1, 0]);  arg849_1 = None
    addmm_230: "f32[128, 128]" = torch.ops.aten.addmm.default(arg850_1, view_610, permute_305);  arg850_1 = view_610 = permute_305 = None
    view_611: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_230, [1, 128, 128]);  addmm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_614: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_611, [1, 128, 4, 32]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_308: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
    
    # No stacktrace found for following nodes
    clone_default_26: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_24, clone_default_25, clone_default_26, scale = 0.17677669529663687);  clone_default_24 = clone_default_25 = clone_default_26 = None
    getitem_10: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_8[0];  _scaled_dot_product_flash_attention_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_310: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_621: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_310, [1, 128, 128]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_622: "f32[128, 128]" = torch.ops.aten.reshape.default(view_621, [128, 128]);  view_621 = None
    permute_311: "f32[128, 128]" = torch.ops.aten.permute.default(arg851_1, [1, 0]);  arg851_1 = None
    addmm_231: "f32[128, 128]" = torch.ops.aten.addmm.default(arg852_1, view_622, permute_311);  arg852_1 = view_622 = permute_311 = None
    view_623: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_231, [1, 128, 128]);  addmm_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_602: "f32[128, 512]" = torch.ops.aten.reshape.default(add_227, [128, 512])
    permute_301: "f32[512, 128]" = torch.ops.aten.permute.default(arg841_1, [1, 0]);  arg841_1 = None
    addmm_226: "f32[128, 128]" = torch.ops.aten.addmm.default(arg842_1, view_602, permute_301);  arg842_1 = view_602 = permute_301 = None
    view_603: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_226, [1, 128, 128]);  addmm_226 = None
    
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
    permute_312: "f32[128, 512]" = torch.ops.aten.permute.default(arg853_1, [1, 0]);  arg853_1 = None
    addmm_232: "f32[128, 512]" = torch.ops.aten.addmm.default(arg854_1, view_624, permute_312);  arg854_1 = view_624 = permute_312 = None
    view_625: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_232, [1, 128, 512]);  addmm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_60: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_625);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_626: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_60, [128, 512]);  relu_60 = None
    permute_313: "f32[512, 128]" = torch.ops.aten.permute.default(arg855_1, [1, 0]);  arg855_1 = None
    addmm_233: "f32[128, 128]" = torch.ops.aten.addmm.default(arg856_1, view_626, permute_313);  arg856_1 = view_626 = permute_313 = None
    view_627: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_233, [1, 128, 128]);  addmm_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_233: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_627, add_232);  view_627 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_125: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_233, arg248_1);  add_233 = arg248_1 = None
    add_234: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_125, arg249_1);  mul_125 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_628: "f32[128, 128]" = torch.ops.aten.reshape.default(add_234, [128, 128])
    permute_314: "f32[128, 512]" = torch.ops.aten.permute.default(arg857_1, [1, 0]);  arg857_1 = None
    addmm_234: "f32[128, 512]" = torch.ops.aten.addmm.default(arg858_1, view_628, permute_314);  arg858_1 = view_628 = permute_314 = None
    view_629: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_234, [1, 128, 512]);  addmm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_61: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_629);  view_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_630: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_61, [128, 512]);  relu_61 = None
    permute_315: "f32[512, 128]" = torch.ops.aten.permute.default(arg859_1, [1, 0]);  arg859_1 = None
    addmm_235: "f32[128, 128]" = torch.ops.aten.addmm.default(arg860_1, view_630, permute_315);  arg860_1 = view_630 = permute_315 = None
    view_631: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_235, [1, 128, 128]);  addmm_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_235: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_631, add_234);  view_631 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_126: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_235, arg250_1);  add_235 = arg250_1 = None
    add_236: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_126, arg251_1);  mul_126 = arg251_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_632: "f32[128, 128]" = torch.ops.aten.reshape.default(add_236, [128, 128])
    permute_316: "f32[128, 512]" = torch.ops.aten.permute.default(arg861_1, [1, 0]);  arg861_1 = None
    addmm_236: "f32[128, 512]" = torch.ops.aten.addmm.default(arg862_1, view_632, permute_316);  arg862_1 = view_632 = permute_316 = None
    view_633: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_236, [1, 128, 512]);  addmm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_62: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_633);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_634: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_62, [128, 512]);  relu_62 = None
    permute_317: "f32[512, 128]" = torch.ops.aten.permute.default(arg863_1, [1, 0]);  arg863_1 = None
    addmm_237: "f32[128, 128]" = torch.ops.aten.addmm.default(arg864_1, view_634, permute_317);  arg864_1 = view_634 = permute_317 = None
    view_635: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_237, [1, 128, 128]);  addmm_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_237: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_635, add_236);  view_635 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_127: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_237, arg252_1);  add_237 = arg252_1 = None
    add_238: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_127, arg253_1);  mul_127 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_636: "f32[128, 128]" = torch.ops.aten.reshape.default(add_238, [128, 128])
    permute_318: "f32[128, 512]" = torch.ops.aten.permute.default(arg865_1, [1, 0]);  arg865_1 = None
    addmm_238: "f32[128, 512]" = torch.ops.aten.addmm.default(arg866_1, view_636, permute_318);  arg866_1 = view_636 = permute_318 = None
    view_637: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_238, [1, 128, 512]);  addmm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_63: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_637);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_638: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_63, [128, 512]);  relu_63 = None
    permute_319: "f32[512, 128]" = torch.ops.aten.permute.default(arg867_1, [1, 0]);  arg867_1 = None
    addmm_239: "f32[128, 128]" = torch.ops.aten.addmm.default(arg868_1, view_638, permute_319);  arg868_1 = view_638 = permute_319 = None
    view_639: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_239, [1, 128, 128]);  addmm_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_239: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_639, add_238);  view_639 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_128: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_239, arg254_1);  add_239 = arg254_1 = None
    add_240: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, arg255_1);  mul_128 = arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_640: "f32[128, 128]" = torch.ops.aten.reshape.default(add_240, [128, 128]);  add_240 = None
    permute_320: "f32[128, 512]" = torch.ops.aten.permute.default(arg869_1, [1, 0]);  arg869_1 = None
    addmm_240: "f32[128, 512]" = torch.ops.aten.addmm.default(arg870_1, view_640, permute_320);  arg870_1 = view_640 = permute_320 = None
    view_641: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_240, [1, 128, 512]);  addmm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_241: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_641, add_227);  view_641 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_129: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_241, arg256_1);  add_241 = arg256_1 = None
    add_242: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_129, arg257_1);  mul_129 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_644: "f32[128, 512]" = torch.ops.aten.reshape.default(add_242, [128, 512])
    permute_322: "f32[512, 128]" = torch.ops.aten.permute.default(arg873_1, [1, 0]);  arg873_1 = None
    addmm_242: "f32[128, 128]" = torch.ops.aten.addmm.default(arg874_1, view_644, permute_322);  arg874_1 = view_644 = permute_322 = None
    view_645: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_242, [1, 128, 128]);  addmm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_131: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_645, arg260_1);  view_645 = arg260_1 = None
    add_244: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_131, arg261_1);  mul_131 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_646: "f32[128, 128]" = torch.ops.aten.reshape.default(add_244, [128, 128])
    permute_323: "f32[128, 128]" = torch.ops.aten.permute.default(arg875_1, [1, 0]);  arg875_1 = None
    addmm_243: "f32[128, 128]" = torch.ops.aten.addmm.default(arg876_1, view_646, permute_323);  arg876_1 = view_646 = permute_323 = None
    view_647: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_243, [1, 128, 128]);  addmm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_652: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_647, [1, 128, 4, 32]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
    
    # No stacktrace found for following nodes
    clone_default_21: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_648: "f32[128, 128]" = torch.ops.aten.reshape.default(add_244, [128, 128]);  add_244 = None
    permute_324: "f32[128, 128]" = torch.ops.aten.permute.default(arg877_1, [1, 0]);  arg877_1 = None
    addmm_244: "f32[128, 128]" = torch.ops.aten.addmm.default(arg878_1, view_648, permute_324);  arg878_1 = view_648 = permute_324 = None
    view_649: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_244, [1, 128, 128]);  addmm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_653: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_649, [1, 128, 4, 32]);  view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    
    # No stacktrace found for following nodes
    clone_default_22: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_650: "f32[128, 512]" = torch.ops.aten.reshape.default(add_242, [128, 512])
    permute_325: "f32[512, 128]" = torch.ops.aten.permute.default(arg879_1, [1, 0]);  arg879_1 = None
    addmm_245: "f32[128, 128]" = torch.ops.aten.addmm.default(arg880_1, view_650, permute_325);  arg880_1 = view_650 = permute_325 = None
    view_651: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_245, [1, 128, 128]);  addmm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_654: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_651, [1, 128, 4, 32]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_328: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # No stacktrace found for following nodes
    clone_default_23: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_21, clone_default_22, clone_default_23, scale = 0.17677669529663687);  clone_default_21 = clone_default_22 = clone_default_23 = None
    getitem_9: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_7[0];  _scaled_dot_product_flash_attention_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_330: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_661: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_330, [1, 128, 128]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_662: "f32[128, 128]" = torch.ops.aten.reshape.default(view_661, [128, 128]);  view_661 = None
    permute_331: "f32[128, 128]" = torch.ops.aten.permute.default(arg881_1, [1, 0]);  arg881_1 = None
    addmm_246: "f32[128, 128]" = torch.ops.aten.addmm.default(arg882_1, view_662, permute_331);  arg882_1 = view_662 = permute_331 = None
    view_663: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_246, [1, 128, 128]);  addmm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_642: "f32[128, 512]" = torch.ops.aten.reshape.default(add_242, [128, 512])
    permute_321: "f32[512, 128]" = torch.ops.aten.permute.default(arg871_1, [1, 0]);  arg871_1 = None
    addmm_241: "f32[128, 128]" = torch.ops.aten.addmm.default(arg872_1, view_642, permute_321);  arg872_1 = view_642 = permute_321 = None
    view_643: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_241, [1, 128, 128]);  addmm_241 = None
    
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
    permute_332: "f32[128, 512]" = torch.ops.aten.permute.default(arg883_1, [1, 0]);  arg883_1 = None
    addmm_247: "f32[128, 512]" = torch.ops.aten.addmm.default(arg884_1, view_664, permute_332);  arg884_1 = view_664 = permute_332 = None
    view_665: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_247, [1, 128, 512]);  addmm_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_64: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_665);  view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_666: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_64, [128, 512]);  relu_64 = None
    permute_333: "f32[512, 128]" = torch.ops.aten.permute.default(arg885_1, [1, 0]);  arg885_1 = None
    addmm_248: "f32[128, 128]" = torch.ops.aten.addmm.default(arg886_1, view_666, permute_333);  arg886_1 = view_666 = permute_333 = None
    view_667: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_248, [1, 128, 128]);  addmm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_248: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_667, add_247);  view_667 = add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_133: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_248, arg264_1);  add_248 = arg264_1 = None
    add_249: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_133, arg265_1);  mul_133 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_668: "f32[128, 128]" = torch.ops.aten.reshape.default(add_249, [128, 128])
    permute_334: "f32[128, 512]" = torch.ops.aten.permute.default(arg887_1, [1, 0]);  arg887_1 = None
    addmm_249: "f32[128, 512]" = torch.ops.aten.addmm.default(arg888_1, view_668, permute_334);  arg888_1 = view_668 = permute_334 = None
    view_669: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_249, [1, 128, 512]);  addmm_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_65: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_669);  view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_670: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_65, [128, 512]);  relu_65 = None
    permute_335: "f32[512, 128]" = torch.ops.aten.permute.default(arg889_1, [1, 0]);  arg889_1 = None
    addmm_250: "f32[128, 128]" = torch.ops.aten.addmm.default(arg890_1, view_670, permute_335);  arg890_1 = view_670 = permute_335 = None
    view_671: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_250, [1, 128, 128]);  addmm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_250: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_671, add_249);  view_671 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_134: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_250, arg266_1);  add_250 = arg266_1 = None
    add_251: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_134, arg267_1);  mul_134 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_672: "f32[128, 128]" = torch.ops.aten.reshape.default(add_251, [128, 128])
    permute_336: "f32[128, 512]" = torch.ops.aten.permute.default(arg891_1, [1, 0]);  arg891_1 = None
    addmm_251: "f32[128, 512]" = torch.ops.aten.addmm.default(arg892_1, view_672, permute_336);  arg892_1 = view_672 = permute_336 = None
    view_673: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_251, [1, 128, 512]);  addmm_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_66: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_673);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_674: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_66, [128, 512]);  relu_66 = None
    permute_337: "f32[512, 128]" = torch.ops.aten.permute.default(arg893_1, [1, 0]);  arg893_1 = None
    addmm_252: "f32[128, 128]" = torch.ops.aten.addmm.default(arg894_1, view_674, permute_337);  arg894_1 = view_674 = permute_337 = None
    view_675: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_252, [1, 128, 128]);  addmm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_252: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_675, add_251);  view_675 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_135: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_252, arg268_1);  add_252 = arg268_1 = None
    add_253: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_135, arg269_1);  mul_135 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_676: "f32[128, 128]" = torch.ops.aten.reshape.default(add_253, [128, 128])
    permute_338: "f32[128, 512]" = torch.ops.aten.permute.default(arg895_1, [1, 0]);  arg895_1 = None
    addmm_253: "f32[128, 512]" = torch.ops.aten.addmm.default(arg896_1, view_676, permute_338);  arg896_1 = view_676 = permute_338 = None
    view_677: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_253, [1, 128, 512]);  addmm_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_67: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_677);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_678: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_67, [128, 512]);  relu_67 = None
    permute_339: "f32[512, 128]" = torch.ops.aten.permute.default(arg897_1, [1, 0]);  arg897_1 = None
    addmm_254: "f32[128, 128]" = torch.ops.aten.addmm.default(arg898_1, view_678, permute_339);  arg898_1 = view_678 = permute_339 = None
    view_679: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_254, [1, 128, 128]);  addmm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_254: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_679, add_253);  view_679 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_136: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_254, arg270_1);  add_254 = arg270_1 = None
    add_255: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_136, arg271_1);  mul_136 = arg271_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_680: "f32[128, 128]" = torch.ops.aten.reshape.default(add_255, [128, 128]);  add_255 = None
    permute_340: "f32[128, 512]" = torch.ops.aten.permute.default(arg899_1, [1, 0]);  arg899_1 = None
    addmm_255: "f32[128, 512]" = torch.ops.aten.addmm.default(arg900_1, view_680, permute_340);  arg900_1 = view_680 = permute_340 = None
    view_681: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_255, [1, 128, 512]);  addmm_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_256: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_681, add_242);  view_681 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_137: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_256, arg272_1);  add_256 = arg272_1 = None
    add_257: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_137, arg273_1);  mul_137 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_684: "f32[128, 512]" = torch.ops.aten.reshape.default(add_257, [128, 512])
    permute_342: "f32[512, 128]" = torch.ops.aten.permute.default(arg903_1, [1, 0]);  arg903_1 = None
    addmm_257: "f32[128, 128]" = torch.ops.aten.addmm.default(arg904_1, view_684, permute_342);  arg904_1 = view_684 = permute_342 = None
    view_685: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_257, [1, 128, 128]);  addmm_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_139: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_685, arg276_1);  view_685 = arg276_1 = None
    add_259: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_139, arg277_1);  mul_139 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_686: "f32[128, 128]" = torch.ops.aten.reshape.default(add_259, [128, 128])
    permute_343: "f32[128, 128]" = torch.ops.aten.permute.default(arg905_1, [1, 0]);  arg905_1 = None
    addmm_258: "f32[128, 128]" = torch.ops.aten.addmm.default(arg906_1, view_686, permute_343);  arg906_1 = view_686 = permute_343 = None
    view_687: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_258, [1, 128, 128]);  addmm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_692: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_687, [1, 128, 4, 32]);  view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_346: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
    
    # No stacktrace found for following nodes
    clone_default_18: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_688: "f32[128, 128]" = torch.ops.aten.reshape.default(add_259, [128, 128]);  add_259 = None
    permute_344: "f32[128, 128]" = torch.ops.aten.permute.default(arg907_1, [1, 0]);  arg907_1 = None
    addmm_259: "f32[128, 128]" = torch.ops.aten.addmm.default(arg908_1, view_688, permute_344);  arg908_1 = view_688 = permute_344 = None
    view_689: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_259, [1, 128, 128]);  addmm_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_693: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_689, [1, 128, 4, 32]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_347: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
    
    # No stacktrace found for following nodes
    clone_default_19: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_690: "f32[128, 512]" = torch.ops.aten.reshape.default(add_257, [128, 512])
    permute_345: "f32[512, 128]" = torch.ops.aten.permute.default(arg909_1, [1, 0]);  arg909_1 = None
    addmm_260: "f32[128, 128]" = torch.ops.aten.addmm.default(arg910_1, view_690, permute_345);  arg910_1 = view_690 = permute_345 = None
    view_691: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_260, [1, 128, 128]);  addmm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_694: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_691, [1, 128, 4, 32]);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_348: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
    
    # No stacktrace found for following nodes
    clone_default_20: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_18, clone_default_19, clone_default_20, scale = 0.17677669529663687);  clone_default_18 = clone_default_19 = clone_default_20 = None
    getitem_8: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_6[0];  _scaled_dot_product_flash_attention_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_350: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_701: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_350, [1, 128, 128]);  permute_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_702: "f32[128, 128]" = torch.ops.aten.reshape.default(view_701, [128, 128]);  view_701 = None
    permute_351: "f32[128, 128]" = torch.ops.aten.permute.default(arg911_1, [1, 0]);  arg911_1 = None
    addmm_261: "f32[128, 128]" = torch.ops.aten.addmm.default(arg912_1, view_702, permute_351);  arg912_1 = view_702 = permute_351 = None
    view_703: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_261, [1, 128, 128]);  addmm_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_682: "f32[128, 512]" = torch.ops.aten.reshape.default(add_257, [128, 512])
    permute_341: "f32[512, 128]" = torch.ops.aten.permute.default(arg901_1, [1, 0]);  arg901_1 = None
    addmm_256: "f32[128, 128]" = torch.ops.aten.addmm.default(arg902_1, view_682, permute_341);  arg902_1 = view_682 = permute_341 = None
    view_683: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_256, [1, 128, 128]);  addmm_256 = None
    
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
    permute_352: "f32[128, 512]" = torch.ops.aten.permute.default(arg913_1, [1, 0]);  arg913_1 = None
    addmm_262: "f32[128, 512]" = torch.ops.aten.addmm.default(arg914_1, view_704, permute_352);  arg914_1 = view_704 = permute_352 = None
    view_705: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_262, [1, 128, 512]);  addmm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_68: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_705);  view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_706: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_68, [128, 512]);  relu_68 = None
    permute_353: "f32[512, 128]" = torch.ops.aten.permute.default(arg915_1, [1, 0]);  arg915_1 = None
    addmm_263: "f32[128, 128]" = torch.ops.aten.addmm.default(arg916_1, view_706, permute_353);  arg916_1 = view_706 = permute_353 = None
    view_707: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_263, [1, 128, 128]);  addmm_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_263: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_707, add_262);  view_707 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_141: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_263, arg280_1);  add_263 = arg280_1 = None
    add_264: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_141, arg281_1);  mul_141 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_708: "f32[128, 128]" = torch.ops.aten.reshape.default(add_264, [128, 128])
    permute_354: "f32[128, 512]" = torch.ops.aten.permute.default(arg917_1, [1, 0]);  arg917_1 = None
    addmm_264: "f32[128, 512]" = torch.ops.aten.addmm.default(arg918_1, view_708, permute_354);  arg918_1 = view_708 = permute_354 = None
    view_709: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_264, [1, 128, 512]);  addmm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_69: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_709);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_710: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_69, [128, 512]);  relu_69 = None
    permute_355: "f32[512, 128]" = torch.ops.aten.permute.default(arg919_1, [1, 0]);  arg919_1 = None
    addmm_265: "f32[128, 128]" = torch.ops.aten.addmm.default(arg920_1, view_710, permute_355);  arg920_1 = view_710 = permute_355 = None
    view_711: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_265, [1, 128, 128]);  addmm_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_265: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_711, add_264);  view_711 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_142: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_265, arg282_1);  add_265 = arg282_1 = None
    add_266: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_142, arg283_1);  mul_142 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_712: "f32[128, 128]" = torch.ops.aten.reshape.default(add_266, [128, 128])
    permute_356: "f32[128, 512]" = torch.ops.aten.permute.default(arg921_1, [1, 0]);  arg921_1 = None
    addmm_266: "f32[128, 512]" = torch.ops.aten.addmm.default(arg922_1, view_712, permute_356);  arg922_1 = view_712 = permute_356 = None
    view_713: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_266, [1, 128, 512]);  addmm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_70: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_713);  view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_714: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_70, [128, 512]);  relu_70 = None
    permute_357: "f32[512, 128]" = torch.ops.aten.permute.default(arg923_1, [1, 0]);  arg923_1 = None
    addmm_267: "f32[128, 128]" = torch.ops.aten.addmm.default(arg924_1, view_714, permute_357);  arg924_1 = view_714 = permute_357 = None
    view_715: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_267, [1, 128, 128]);  addmm_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_267: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_715, add_266);  view_715 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_143: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_267, arg284_1);  add_267 = arg284_1 = None
    add_268: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_143, arg285_1);  mul_143 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_716: "f32[128, 128]" = torch.ops.aten.reshape.default(add_268, [128, 128])
    permute_358: "f32[128, 512]" = torch.ops.aten.permute.default(arg925_1, [1, 0]);  arg925_1 = None
    addmm_268: "f32[128, 512]" = torch.ops.aten.addmm.default(arg926_1, view_716, permute_358);  arg926_1 = view_716 = permute_358 = None
    view_717: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_268, [1, 128, 512]);  addmm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_71: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_717);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_718: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_71, [128, 512]);  relu_71 = None
    permute_359: "f32[512, 128]" = torch.ops.aten.permute.default(arg927_1, [1, 0]);  arg927_1 = None
    addmm_269: "f32[128, 128]" = torch.ops.aten.addmm.default(arg928_1, view_718, permute_359);  arg928_1 = view_718 = permute_359 = None
    view_719: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_269, [1, 128, 128]);  addmm_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_269: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_719, add_268);  view_719 = add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_144: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_269, arg286_1);  add_269 = arg286_1 = None
    add_270: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_144, arg287_1);  mul_144 = arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_720: "f32[128, 128]" = torch.ops.aten.reshape.default(add_270, [128, 128]);  add_270 = None
    permute_360: "f32[128, 512]" = torch.ops.aten.permute.default(arg929_1, [1, 0]);  arg929_1 = None
    addmm_270: "f32[128, 512]" = torch.ops.aten.addmm.default(arg930_1, view_720, permute_360);  arg930_1 = view_720 = permute_360 = None
    view_721: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_270, [1, 128, 512]);  addmm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_271: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_721, add_257);  view_721 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_145: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_271, arg288_1);  add_271 = arg288_1 = None
    add_272: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_145, arg289_1);  mul_145 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_724: "f32[128, 512]" = torch.ops.aten.reshape.default(add_272, [128, 512])
    permute_362: "f32[512, 128]" = torch.ops.aten.permute.default(arg933_1, [1, 0]);  arg933_1 = None
    addmm_272: "f32[128, 128]" = torch.ops.aten.addmm.default(arg934_1, view_724, permute_362);  arg934_1 = view_724 = permute_362 = None
    view_725: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_272, [1, 128, 128]);  addmm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_147: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_725, arg292_1);  view_725 = arg292_1 = None
    add_274: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_147, arg293_1);  mul_147 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_726: "f32[128, 128]" = torch.ops.aten.reshape.default(add_274, [128, 128])
    permute_363: "f32[128, 128]" = torch.ops.aten.permute.default(arg935_1, [1, 0]);  arg935_1 = None
    addmm_273: "f32[128, 128]" = torch.ops.aten.addmm.default(arg936_1, view_726, permute_363);  arg936_1 = view_726 = permute_363 = None
    view_727: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_273, [1, 128, 128]);  addmm_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_732: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_727, [1, 128, 4, 32]);  view_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_366: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
    
    # No stacktrace found for following nodes
    clone_default_15: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_728: "f32[128, 128]" = torch.ops.aten.reshape.default(add_274, [128, 128]);  add_274 = None
    permute_364: "f32[128, 128]" = torch.ops.aten.permute.default(arg937_1, [1, 0]);  arg937_1 = None
    addmm_274: "f32[128, 128]" = torch.ops.aten.addmm.default(arg938_1, view_728, permute_364);  arg938_1 = view_728 = permute_364 = None
    view_729: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_274, [1, 128, 128]);  addmm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_733: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_729, [1, 128, 4, 32]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_367: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_733, [0, 2, 1, 3]);  view_733 = None
    
    # No stacktrace found for following nodes
    clone_default_16: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_730: "f32[128, 512]" = torch.ops.aten.reshape.default(add_272, [128, 512])
    permute_365: "f32[512, 128]" = torch.ops.aten.permute.default(arg939_1, [1, 0]);  arg939_1 = None
    addmm_275: "f32[128, 128]" = torch.ops.aten.addmm.default(arg940_1, view_730, permute_365);  arg940_1 = view_730 = permute_365 = None
    view_731: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_275, [1, 128, 128]);  addmm_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_734: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_731, [1, 128, 4, 32]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_368: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
    
    # No stacktrace found for following nodes
    clone_default_17: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_15, clone_default_16, clone_default_17, scale = 0.17677669529663687);  clone_default_15 = clone_default_16 = clone_default_17 = None
    getitem_7: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_5[0];  _scaled_dot_product_flash_attention_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_370: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_7, [0, 2, 1, 3]);  getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_741: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_370, [1, 128, 128]);  permute_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_742: "f32[128, 128]" = torch.ops.aten.reshape.default(view_741, [128, 128]);  view_741 = None
    permute_371: "f32[128, 128]" = torch.ops.aten.permute.default(arg941_1, [1, 0]);  arg941_1 = None
    addmm_276: "f32[128, 128]" = torch.ops.aten.addmm.default(arg942_1, view_742, permute_371);  arg942_1 = view_742 = permute_371 = None
    view_743: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_276, [1, 128, 128]);  addmm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_722: "f32[128, 512]" = torch.ops.aten.reshape.default(add_272, [128, 512])
    permute_361: "f32[512, 128]" = torch.ops.aten.permute.default(arg931_1, [1, 0]);  arg931_1 = None
    addmm_271: "f32[128, 128]" = torch.ops.aten.addmm.default(arg932_1, view_722, permute_361);  arg932_1 = view_722 = permute_361 = None
    view_723: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_271, [1, 128, 128]);  addmm_271 = None
    
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
    permute_372: "f32[128, 512]" = torch.ops.aten.permute.default(arg943_1, [1, 0]);  arg943_1 = None
    addmm_277: "f32[128, 512]" = torch.ops.aten.addmm.default(arg944_1, view_744, permute_372);  arg944_1 = view_744 = permute_372 = None
    view_745: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_277, [1, 128, 512]);  addmm_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_72: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_745);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_746: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_72, [128, 512]);  relu_72 = None
    permute_373: "f32[512, 128]" = torch.ops.aten.permute.default(arg945_1, [1, 0]);  arg945_1 = None
    addmm_278: "f32[128, 128]" = torch.ops.aten.addmm.default(arg946_1, view_746, permute_373);  arg946_1 = view_746 = permute_373 = None
    view_747: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_278, [1, 128, 128]);  addmm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_278: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_747, add_277);  view_747 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_149: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_278, arg296_1);  add_278 = arg296_1 = None
    add_279: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_149, arg297_1);  mul_149 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_748: "f32[128, 128]" = torch.ops.aten.reshape.default(add_279, [128, 128])
    permute_374: "f32[128, 512]" = torch.ops.aten.permute.default(arg947_1, [1, 0]);  arg947_1 = None
    addmm_279: "f32[128, 512]" = torch.ops.aten.addmm.default(arg948_1, view_748, permute_374);  arg948_1 = view_748 = permute_374 = None
    view_749: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_279, [1, 128, 512]);  addmm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_73: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_749);  view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_750: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_73, [128, 512]);  relu_73 = None
    permute_375: "f32[512, 128]" = torch.ops.aten.permute.default(arg949_1, [1, 0]);  arg949_1 = None
    addmm_280: "f32[128, 128]" = torch.ops.aten.addmm.default(arg950_1, view_750, permute_375);  arg950_1 = view_750 = permute_375 = None
    view_751: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_280, [1, 128, 128]);  addmm_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_280: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_751, add_279);  view_751 = add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_150: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_280, arg298_1);  add_280 = arg298_1 = None
    add_281: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_150, arg299_1);  mul_150 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_752: "f32[128, 128]" = torch.ops.aten.reshape.default(add_281, [128, 128])
    permute_376: "f32[128, 512]" = torch.ops.aten.permute.default(arg951_1, [1, 0]);  arg951_1 = None
    addmm_281: "f32[128, 512]" = torch.ops.aten.addmm.default(arg952_1, view_752, permute_376);  arg952_1 = view_752 = permute_376 = None
    view_753: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_281, [1, 128, 512]);  addmm_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_74: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_753);  view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_754: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_74, [128, 512]);  relu_74 = None
    permute_377: "f32[512, 128]" = torch.ops.aten.permute.default(arg953_1, [1, 0]);  arg953_1 = None
    addmm_282: "f32[128, 128]" = torch.ops.aten.addmm.default(arg954_1, view_754, permute_377);  arg954_1 = view_754 = permute_377 = None
    view_755: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_282, [1, 128, 128]);  addmm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_282: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_755, add_281);  view_755 = add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_151: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_282, arg300_1);  add_282 = arg300_1 = None
    add_283: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_151, arg301_1);  mul_151 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_756: "f32[128, 128]" = torch.ops.aten.reshape.default(add_283, [128, 128])
    permute_378: "f32[128, 512]" = torch.ops.aten.permute.default(arg955_1, [1, 0]);  arg955_1 = None
    addmm_283: "f32[128, 512]" = torch.ops.aten.addmm.default(arg956_1, view_756, permute_378);  arg956_1 = view_756 = permute_378 = None
    view_757: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_283, [1, 128, 512]);  addmm_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_75: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_757);  view_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_758: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_75, [128, 512]);  relu_75 = None
    permute_379: "f32[512, 128]" = torch.ops.aten.permute.default(arg957_1, [1, 0]);  arg957_1 = None
    addmm_284: "f32[128, 128]" = torch.ops.aten.addmm.default(arg958_1, view_758, permute_379);  arg958_1 = view_758 = permute_379 = None
    view_759: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_284, [1, 128, 128]);  addmm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_284: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_759, add_283);  view_759 = add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_152: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_284, arg302_1);  add_284 = arg302_1 = None
    add_285: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_152, arg303_1);  mul_152 = arg303_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_760: "f32[128, 128]" = torch.ops.aten.reshape.default(add_285, [128, 128]);  add_285 = None
    permute_380: "f32[128, 512]" = torch.ops.aten.permute.default(arg959_1, [1, 0]);  arg959_1 = None
    addmm_285: "f32[128, 512]" = torch.ops.aten.addmm.default(arg960_1, view_760, permute_380);  arg960_1 = view_760 = permute_380 = None
    view_761: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_285, [1, 128, 512]);  addmm_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_286: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_761, add_272);  view_761 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_153: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_286, arg304_1);  add_286 = arg304_1 = None
    add_287: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_153, arg305_1);  mul_153 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_764: "f32[128, 512]" = torch.ops.aten.reshape.default(add_287, [128, 512])
    permute_382: "f32[512, 128]" = torch.ops.aten.permute.default(arg963_1, [1, 0]);  arg963_1 = None
    addmm_287: "f32[128, 128]" = torch.ops.aten.addmm.default(arg964_1, view_764, permute_382);  arg964_1 = view_764 = permute_382 = None
    view_765: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_287, [1, 128, 128]);  addmm_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_155: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_765, arg308_1);  view_765 = arg308_1 = None
    add_289: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_155, arg309_1);  mul_155 = arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_766: "f32[128, 128]" = torch.ops.aten.reshape.default(add_289, [128, 128])
    permute_383: "f32[128, 128]" = torch.ops.aten.permute.default(arg965_1, [1, 0]);  arg965_1 = None
    addmm_288: "f32[128, 128]" = torch.ops.aten.addmm.default(arg966_1, view_766, permute_383);  arg966_1 = view_766 = permute_383 = None
    view_767: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_288, [1, 128, 128]);  addmm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_772: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_767, [1, 128, 4, 32]);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_386: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_768: "f32[128, 128]" = torch.ops.aten.reshape.default(add_289, [128, 128]);  add_289 = None
    permute_384: "f32[128, 128]" = torch.ops.aten.permute.default(arg967_1, [1, 0]);  arg967_1 = None
    addmm_289: "f32[128, 128]" = torch.ops.aten.addmm.default(arg968_1, view_768, permute_384);  arg968_1 = view_768 = permute_384 = None
    view_769: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_289, [1, 128, 128]);  addmm_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_773: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_769, [1, 128, 4, 32]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None
    
    # No stacktrace found for following nodes
    clone_default_13: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_770: "f32[128, 512]" = torch.ops.aten.reshape.default(add_287, [128, 512])
    permute_385: "f32[512, 128]" = torch.ops.aten.permute.default(arg969_1, [1, 0]);  arg969_1 = None
    addmm_290: "f32[128, 128]" = torch.ops.aten.addmm.default(arg970_1, view_770, permute_385);  arg970_1 = view_770 = permute_385 = None
    view_771: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_290, [1, 128, 128]);  addmm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_774: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_771, [1, 128, 4, 32]);  view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_388: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
    
    # No stacktrace found for following nodes
    clone_default_14: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
    _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_12, clone_default_13, clone_default_14, scale = 0.17677669529663687);  clone_default_12 = clone_default_13 = clone_default_14 = None
    getitem_6: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_4[0];  _scaled_dot_product_flash_attention_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_390: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_781: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_390, [1, 128, 128]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_782: "f32[128, 128]" = torch.ops.aten.reshape.default(view_781, [128, 128]);  view_781 = None
    permute_391: "f32[128, 128]" = torch.ops.aten.permute.default(arg971_1, [1, 0]);  arg971_1 = None
    addmm_291: "f32[128, 128]" = torch.ops.aten.addmm.default(arg972_1, view_782, permute_391);  arg972_1 = view_782 = permute_391 = None
    view_783: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_291, [1, 128, 128]);  addmm_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_762: "f32[128, 512]" = torch.ops.aten.reshape.default(add_287, [128, 512])
    permute_381: "f32[512, 128]" = torch.ops.aten.permute.default(arg961_1, [1, 0]);  arg961_1 = None
    addmm_286: "f32[128, 128]" = torch.ops.aten.addmm.default(arg962_1, view_762, permute_381);  arg962_1 = view_762 = permute_381 = None
    view_763: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_286, [1, 128, 128]);  addmm_286 = None
    
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
    permute_392: "f32[128, 512]" = torch.ops.aten.permute.default(arg973_1, [1, 0]);  arg973_1 = None
    addmm_292: "f32[128, 512]" = torch.ops.aten.addmm.default(arg974_1, view_784, permute_392);  arg974_1 = view_784 = permute_392 = None
    view_785: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_292, [1, 128, 512]);  addmm_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_76: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_785);  view_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_786: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_76, [128, 512]);  relu_76 = None
    permute_393: "f32[512, 128]" = torch.ops.aten.permute.default(arg975_1, [1, 0]);  arg975_1 = None
    addmm_293: "f32[128, 128]" = torch.ops.aten.addmm.default(arg976_1, view_786, permute_393);  arg976_1 = view_786 = permute_393 = None
    view_787: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_293, [1, 128, 128]);  addmm_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_293: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_787, add_292);  view_787 = add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_157: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, arg312_1);  add_293 = arg312_1 = None
    add_294: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_157, arg313_1);  mul_157 = arg313_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_788: "f32[128, 128]" = torch.ops.aten.reshape.default(add_294, [128, 128])
    permute_394: "f32[128, 512]" = torch.ops.aten.permute.default(arg977_1, [1, 0]);  arg977_1 = None
    addmm_294: "f32[128, 512]" = torch.ops.aten.addmm.default(arg978_1, view_788, permute_394);  arg978_1 = view_788 = permute_394 = None
    view_789: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_294, [1, 128, 512]);  addmm_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_77: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_789);  view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_790: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_77, [128, 512]);  relu_77 = None
    permute_395: "f32[512, 128]" = torch.ops.aten.permute.default(arg979_1, [1, 0]);  arg979_1 = None
    addmm_295: "f32[128, 128]" = torch.ops.aten.addmm.default(arg980_1, view_790, permute_395);  arg980_1 = view_790 = permute_395 = None
    view_791: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_295, [1, 128, 128]);  addmm_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_295: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_791, add_294);  view_791 = add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_158: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_295, arg314_1);  add_295 = arg314_1 = None
    add_296: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_158, arg315_1);  mul_158 = arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_792: "f32[128, 128]" = torch.ops.aten.reshape.default(add_296, [128, 128])
    permute_396: "f32[128, 512]" = torch.ops.aten.permute.default(arg981_1, [1, 0]);  arg981_1 = None
    addmm_296: "f32[128, 512]" = torch.ops.aten.addmm.default(arg982_1, view_792, permute_396);  arg982_1 = view_792 = permute_396 = None
    view_793: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_296, [1, 128, 512]);  addmm_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_78: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_793);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_794: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_78, [128, 512]);  relu_78 = None
    permute_397: "f32[512, 128]" = torch.ops.aten.permute.default(arg983_1, [1, 0]);  arg983_1 = None
    addmm_297: "f32[128, 128]" = torch.ops.aten.addmm.default(arg984_1, view_794, permute_397);  arg984_1 = view_794 = permute_397 = None
    view_795: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_297, [1, 128, 128]);  addmm_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_297: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_795, add_296);  view_795 = add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_159: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_297, arg316_1);  add_297 = arg316_1 = None
    add_298: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_159, arg317_1);  mul_159 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_796: "f32[128, 128]" = torch.ops.aten.reshape.default(add_298, [128, 128])
    permute_398: "f32[128, 512]" = torch.ops.aten.permute.default(arg985_1, [1, 0]);  arg985_1 = None
    addmm_298: "f32[128, 512]" = torch.ops.aten.addmm.default(arg986_1, view_796, permute_398);  arg986_1 = view_796 = permute_398 = None
    view_797: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_298, [1, 128, 512]);  addmm_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_79: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_797);  view_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_798: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_79, [128, 512]);  relu_79 = None
    permute_399: "f32[512, 128]" = torch.ops.aten.permute.default(arg987_1, [1, 0]);  arg987_1 = None
    addmm_299: "f32[128, 128]" = torch.ops.aten.addmm.default(arg988_1, view_798, permute_399);  arg988_1 = view_798 = permute_399 = None
    view_799: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_299, [1, 128, 128]);  addmm_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_299: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_799, add_298);  view_799 = add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_160: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_299, arg318_1);  add_299 = arg318_1 = None
    add_300: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_160, arg319_1);  mul_160 = arg319_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_800: "f32[128, 128]" = torch.ops.aten.reshape.default(add_300, [128, 128]);  add_300 = None
    permute_400: "f32[128, 512]" = torch.ops.aten.permute.default(arg989_1, [1, 0]);  arg989_1 = None
    addmm_300: "f32[128, 512]" = torch.ops.aten.addmm.default(arg990_1, view_800, permute_400);  arg990_1 = view_800 = permute_400 = None
    view_801: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_300, [1, 128, 512]);  addmm_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_301: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_801, add_287);  view_801 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_161: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_301, arg320_1);  add_301 = arg320_1 = None
    add_302: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_161, arg321_1);  mul_161 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_804: "f32[128, 512]" = torch.ops.aten.reshape.default(add_302, [128, 512])
    permute_402: "f32[512, 128]" = torch.ops.aten.permute.default(arg993_1, [1, 0]);  arg993_1 = None
    addmm_302: "f32[128, 128]" = torch.ops.aten.addmm.default(arg994_1, view_804, permute_402);  arg994_1 = view_804 = permute_402 = None
    view_805: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_302, [1, 128, 128]);  addmm_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_163: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_805, arg324_1);  view_805 = arg324_1 = None
    add_304: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_163, arg325_1);  mul_163 = arg325_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_806: "f32[128, 128]" = torch.ops.aten.reshape.default(add_304, [128, 128])
    permute_403: "f32[128, 128]" = torch.ops.aten.permute.default(arg995_1, [1, 0]);  arg995_1 = None
    addmm_303: "f32[128, 128]" = torch.ops.aten.addmm.default(arg996_1, view_806, permute_403);  arg996_1 = view_806 = permute_403 = None
    view_807: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_303, [1, 128, 128]);  addmm_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_812: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_807, [1, 128, 4, 32]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_406: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_812, [0, 2, 1, 3]);  view_812 = None
    
    # No stacktrace found for following nodes
    clone_default_9: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_406, memory_format = torch.contiguous_format);  permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_808: "f32[128, 128]" = torch.ops.aten.reshape.default(add_304, [128, 128]);  add_304 = None
    permute_404: "f32[128, 128]" = torch.ops.aten.permute.default(arg997_1, [1, 0]);  arg997_1 = None
    addmm_304: "f32[128, 128]" = torch.ops.aten.addmm.default(arg998_1, view_808, permute_404);  arg998_1 = view_808 = permute_404 = None
    view_809: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_304, [1, 128, 128]);  addmm_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_813: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_809, [1, 128, 4, 32]);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_407: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_813, [0, 2, 1, 3]);  view_813 = None
    
    # No stacktrace found for following nodes
    clone_default_10: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_810: "f32[128, 512]" = torch.ops.aten.reshape.default(add_302, [128, 512])
    permute_405: "f32[512, 128]" = torch.ops.aten.permute.default(arg999_1, [1, 0]);  arg999_1 = None
    addmm_305: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1000_1, view_810, permute_405);  arg1000_1 = view_810 = permute_405 = None
    view_811: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_305, [1, 128, 128]);  addmm_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_814: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_811, [1, 128, 4, 32]);  view_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_408: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None
    
    # No stacktrace found for following nodes
    clone_default_11: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
    _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_9, clone_default_10, clone_default_11, scale = 0.17677669529663687);  clone_default_9 = clone_default_10 = clone_default_11 = None
    getitem_5: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_3[0];  _scaled_dot_product_flash_attention_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_410: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_821: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_410, [1, 128, 128]);  permute_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_822: "f32[128, 128]" = torch.ops.aten.reshape.default(view_821, [128, 128]);  view_821 = None
    permute_411: "f32[128, 128]" = torch.ops.aten.permute.default(arg1001_1, [1, 0]);  arg1001_1 = None
    addmm_306: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1002_1, view_822, permute_411);  arg1002_1 = view_822 = permute_411 = None
    view_823: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_306, [1, 128, 128]);  addmm_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_802: "f32[128, 512]" = torch.ops.aten.reshape.default(add_302, [128, 512])
    permute_401: "f32[512, 128]" = torch.ops.aten.permute.default(arg991_1, [1, 0]);  arg991_1 = None
    addmm_301: "f32[128, 128]" = torch.ops.aten.addmm.default(arg992_1, view_802, permute_401);  arg992_1 = view_802 = permute_401 = None
    view_803: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_301, [1, 128, 128]);  addmm_301 = None
    
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
    permute_412: "f32[128, 512]" = torch.ops.aten.permute.default(arg1003_1, [1, 0]);  arg1003_1 = None
    addmm_307: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1004_1, view_824, permute_412);  arg1004_1 = view_824 = permute_412 = None
    view_825: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_307, [1, 128, 512]);  addmm_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_80: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_825);  view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_826: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_80, [128, 512]);  relu_80 = None
    permute_413: "f32[512, 128]" = torch.ops.aten.permute.default(arg1005_1, [1, 0]);  arg1005_1 = None
    addmm_308: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1006_1, view_826, permute_413);  arg1006_1 = view_826 = permute_413 = None
    view_827: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_308, [1, 128, 128]);  addmm_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_308: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_827, add_307);  view_827 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_165: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_308, arg328_1);  add_308 = arg328_1 = None
    add_309: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_165, arg329_1);  mul_165 = arg329_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_828: "f32[128, 128]" = torch.ops.aten.reshape.default(add_309, [128, 128])
    permute_414: "f32[128, 512]" = torch.ops.aten.permute.default(arg1007_1, [1, 0]);  arg1007_1 = None
    addmm_309: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1008_1, view_828, permute_414);  arg1008_1 = view_828 = permute_414 = None
    view_829: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_309, [1, 128, 512]);  addmm_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_81: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_829);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_830: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_81, [128, 512]);  relu_81 = None
    permute_415: "f32[512, 128]" = torch.ops.aten.permute.default(arg1009_1, [1, 0]);  arg1009_1 = None
    addmm_310: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1010_1, view_830, permute_415);  arg1010_1 = view_830 = permute_415 = None
    view_831: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_310, [1, 128, 128]);  addmm_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_310: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_831, add_309);  view_831 = add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_166: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_310, arg330_1);  add_310 = arg330_1 = None
    add_311: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_166, arg331_1);  mul_166 = arg331_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_832: "f32[128, 128]" = torch.ops.aten.reshape.default(add_311, [128, 128])
    permute_416: "f32[128, 512]" = torch.ops.aten.permute.default(arg1011_1, [1, 0]);  arg1011_1 = None
    addmm_311: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1012_1, view_832, permute_416);  arg1012_1 = view_832 = permute_416 = None
    view_833: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_311, [1, 128, 512]);  addmm_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_82: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_833);  view_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_834: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_82, [128, 512]);  relu_82 = None
    permute_417: "f32[512, 128]" = torch.ops.aten.permute.default(arg1013_1, [1, 0]);  arg1013_1 = None
    addmm_312: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1014_1, view_834, permute_417);  arg1014_1 = view_834 = permute_417 = None
    view_835: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_312, [1, 128, 128]);  addmm_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_312: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_835, add_311);  view_835 = add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_167: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_312, arg332_1);  add_312 = arg332_1 = None
    add_313: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_167, arg333_1);  mul_167 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_836: "f32[128, 128]" = torch.ops.aten.reshape.default(add_313, [128, 128])
    permute_418: "f32[128, 512]" = torch.ops.aten.permute.default(arg1015_1, [1, 0]);  arg1015_1 = None
    addmm_313: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1016_1, view_836, permute_418);  arg1016_1 = view_836 = permute_418 = None
    view_837: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_313, [1, 128, 512]);  addmm_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_83: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_837);  view_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_838: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_83, [128, 512]);  relu_83 = None
    permute_419: "f32[512, 128]" = torch.ops.aten.permute.default(arg1017_1, [1, 0]);  arg1017_1 = None
    addmm_314: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1018_1, view_838, permute_419);  arg1018_1 = view_838 = permute_419 = None
    view_839: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_314, [1, 128, 128]);  addmm_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_314: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_839, add_313);  view_839 = add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_168: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_314, arg334_1);  add_314 = arg334_1 = None
    add_315: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_168, arg335_1);  mul_168 = arg335_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_840: "f32[128, 128]" = torch.ops.aten.reshape.default(add_315, [128, 128]);  add_315 = None
    permute_420: "f32[128, 512]" = torch.ops.aten.permute.default(arg1019_1, [1, 0]);  arg1019_1 = None
    addmm_315: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1020_1, view_840, permute_420);  arg1020_1 = view_840 = permute_420 = None
    view_841: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_315, [1, 128, 512]);  addmm_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_316: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_841, add_302);  view_841 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_169: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_316, arg336_1);  add_316 = arg336_1 = None
    add_317: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_169, arg337_1);  mul_169 = arg337_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_844: "f32[128, 512]" = torch.ops.aten.reshape.default(add_317, [128, 512])
    permute_422: "f32[512, 128]" = torch.ops.aten.permute.default(arg1023_1, [1, 0]);  arg1023_1 = None
    addmm_317: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1024_1, view_844, permute_422);  arg1024_1 = view_844 = permute_422 = None
    view_845: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_317, [1, 128, 128]);  addmm_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_171: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_845, arg340_1);  view_845 = arg340_1 = None
    add_319: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_171, arg341_1);  mul_171 = arg341_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_846: "f32[128, 128]" = torch.ops.aten.reshape.default(add_319, [128, 128])
    permute_423: "f32[128, 128]" = torch.ops.aten.permute.default(arg1025_1, [1, 0]);  arg1025_1 = None
    addmm_318: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1026_1, view_846, permute_423);  arg1026_1 = view_846 = permute_423 = None
    view_847: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_318, [1, 128, 128]);  addmm_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_852: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_847, [1, 128, 4, 32]);  view_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
    
    # No stacktrace found for following nodes
    clone_default_6: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_848: "f32[128, 128]" = torch.ops.aten.reshape.default(add_319, [128, 128]);  add_319 = None
    permute_424: "f32[128, 128]" = torch.ops.aten.permute.default(arg1027_1, [1, 0]);  arg1027_1 = None
    addmm_319: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1028_1, view_848, permute_424);  arg1028_1 = view_848 = permute_424 = None
    view_849: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_319, [1, 128, 128]);  addmm_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_853: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_849, [1, 128, 4, 32]);  view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_427: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_853, [0, 2, 1, 3]);  view_853 = None
    
    # No stacktrace found for following nodes
    clone_default_7: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_850: "f32[128, 512]" = torch.ops.aten.reshape.default(add_317, [128, 512])
    permute_425: "f32[512, 128]" = torch.ops.aten.permute.default(arg1029_1, [1, 0]);  arg1029_1 = None
    addmm_320: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1030_1, view_850, permute_425);  arg1030_1 = view_850 = permute_425 = None
    view_851: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_320, [1, 128, 128]);  addmm_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_854: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_851, [1, 128, 4, 32]);  view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_428: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None
    
    # No stacktrace found for following nodes
    clone_default_8: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
    _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_6, clone_default_7, clone_default_8, scale = 0.17677669529663687);  clone_default_6 = clone_default_7 = clone_default_8 = None
    getitem_4: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_2[0];  _scaled_dot_product_flash_attention_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_430: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_861: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_430, [1, 128, 128]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_862: "f32[128, 128]" = torch.ops.aten.reshape.default(view_861, [128, 128]);  view_861 = None
    permute_431: "f32[128, 128]" = torch.ops.aten.permute.default(arg1031_1, [1, 0]);  arg1031_1 = None
    addmm_321: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1032_1, view_862, permute_431);  arg1032_1 = view_862 = permute_431 = None
    view_863: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_321, [1, 128, 128]);  addmm_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_842: "f32[128, 512]" = torch.ops.aten.reshape.default(add_317, [128, 512])
    permute_421: "f32[512, 128]" = torch.ops.aten.permute.default(arg1021_1, [1, 0]);  arg1021_1 = None
    addmm_316: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1022_1, view_842, permute_421);  arg1022_1 = view_842 = permute_421 = None
    view_843: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_316, [1, 128, 128]);  addmm_316 = None
    
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
    permute_432: "f32[128, 512]" = torch.ops.aten.permute.default(arg1033_1, [1, 0]);  arg1033_1 = None
    addmm_322: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1034_1, view_864, permute_432);  arg1034_1 = view_864 = permute_432 = None
    view_865: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_322, [1, 128, 512]);  addmm_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_84: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_865);  view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_866: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_84, [128, 512]);  relu_84 = None
    permute_433: "f32[512, 128]" = torch.ops.aten.permute.default(arg1035_1, [1, 0]);  arg1035_1 = None
    addmm_323: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1036_1, view_866, permute_433);  arg1036_1 = view_866 = permute_433 = None
    view_867: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_323, [1, 128, 128]);  addmm_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_323: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_867, add_322);  view_867 = add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_173: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_323, arg344_1);  add_323 = arg344_1 = None
    add_324: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_173, arg345_1);  mul_173 = arg345_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_868: "f32[128, 128]" = torch.ops.aten.reshape.default(add_324, [128, 128])
    permute_434: "f32[128, 512]" = torch.ops.aten.permute.default(arg1037_1, [1, 0]);  arg1037_1 = None
    addmm_324: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1038_1, view_868, permute_434);  arg1038_1 = view_868 = permute_434 = None
    view_869: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_324, [1, 128, 512]);  addmm_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_85: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_869);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_870: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_85, [128, 512]);  relu_85 = None
    permute_435: "f32[512, 128]" = torch.ops.aten.permute.default(arg1039_1, [1, 0]);  arg1039_1 = None
    addmm_325: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1040_1, view_870, permute_435);  arg1040_1 = view_870 = permute_435 = None
    view_871: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_325, [1, 128, 128]);  addmm_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_325: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_871, add_324);  view_871 = add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_174: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_325, arg346_1);  add_325 = arg346_1 = None
    add_326: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_174, arg347_1);  mul_174 = arg347_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_872: "f32[128, 128]" = torch.ops.aten.reshape.default(add_326, [128, 128])
    permute_436: "f32[128, 512]" = torch.ops.aten.permute.default(arg1041_1, [1, 0]);  arg1041_1 = None
    addmm_326: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1042_1, view_872, permute_436);  arg1042_1 = view_872 = permute_436 = None
    view_873: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_326, [1, 128, 512]);  addmm_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_86: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_873);  view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_874: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_86, [128, 512]);  relu_86 = None
    permute_437: "f32[512, 128]" = torch.ops.aten.permute.default(arg1043_1, [1, 0]);  arg1043_1 = None
    addmm_327: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1044_1, view_874, permute_437);  arg1044_1 = view_874 = permute_437 = None
    view_875: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_327, [1, 128, 128]);  addmm_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_327: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_875, add_326);  view_875 = add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_175: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_327, arg348_1);  add_327 = arg348_1 = None
    add_328: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_175, arg349_1);  mul_175 = arg349_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_876: "f32[128, 128]" = torch.ops.aten.reshape.default(add_328, [128, 128])
    permute_438: "f32[128, 512]" = torch.ops.aten.permute.default(arg1045_1, [1, 0]);  arg1045_1 = None
    addmm_328: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1046_1, view_876, permute_438);  arg1046_1 = view_876 = permute_438 = None
    view_877: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_328, [1, 128, 512]);  addmm_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_87: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_877);  view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_878: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_87, [128, 512]);  relu_87 = None
    permute_439: "f32[512, 128]" = torch.ops.aten.permute.default(arg1047_1, [1, 0]);  arg1047_1 = None
    addmm_329: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1048_1, view_878, permute_439);  arg1048_1 = view_878 = permute_439 = None
    view_879: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_329, [1, 128, 128]);  addmm_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_329: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_879, add_328);  view_879 = add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_176: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_329, arg350_1);  add_329 = arg350_1 = None
    add_330: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_176, arg351_1);  mul_176 = arg351_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_880: "f32[128, 128]" = torch.ops.aten.reshape.default(add_330, [128, 128]);  add_330 = None
    permute_440: "f32[128, 512]" = torch.ops.aten.permute.default(arg1049_1, [1, 0]);  arg1049_1 = None
    addmm_330: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1050_1, view_880, permute_440);  arg1050_1 = view_880 = permute_440 = None
    view_881: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_330, [1, 128, 512]);  addmm_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_331: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_881, add_317);  view_881 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_177: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_331, arg352_1);  add_331 = arg352_1 = None
    add_332: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_177, arg353_1);  mul_177 = arg353_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_884: "f32[128, 512]" = torch.ops.aten.reshape.default(add_332, [128, 512])
    permute_442: "f32[512, 128]" = torch.ops.aten.permute.default(arg1053_1, [1, 0]);  arg1053_1 = None
    addmm_332: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1054_1, view_884, permute_442);  arg1054_1 = view_884 = permute_442 = None
    view_885: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_332, [1, 128, 128]);  addmm_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_179: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_885, arg356_1);  view_885 = arg356_1 = None
    add_334: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_179, arg357_1);  mul_179 = arg357_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_886: "f32[128, 128]" = torch.ops.aten.reshape.default(add_334, [128, 128])
    permute_443: "f32[128, 128]" = torch.ops.aten.permute.default(arg1055_1, [1, 0]);  arg1055_1 = None
    addmm_333: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1056_1, view_886, permute_443);  arg1056_1 = view_886 = permute_443 = None
    view_887: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_333, [1, 128, 128]);  addmm_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_892: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_887, [1, 128, 4, 32]);  view_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_446: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
    
    # No stacktrace found for following nodes
    clone_default_3: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_888: "f32[128, 128]" = torch.ops.aten.reshape.default(add_334, [128, 128]);  add_334 = None
    permute_444: "f32[128, 128]" = torch.ops.aten.permute.default(arg1057_1, [1, 0]);  arg1057_1 = None
    addmm_334: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1058_1, view_888, permute_444);  arg1058_1 = view_888 = permute_444 = None
    view_889: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_334, [1, 128, 128]);  addmm_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_893: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_889, [1, 128, 4, 32]);  view_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_447: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None
    
    # No stacktrace found for following nodes
    clone_default_4: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_890: "f32[128, 512]" = torch.ops.aten.reshape.default(add_332, [128, 512])
    permute_445: "f32[512, 128]" = torch.ops.aten.permute.default(arg1059_1, [1, 0]);  arg1059_1 = None
    addmm_335: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1060_1, view_890, permute_445);  arg1060_1 = view_890 = permute_445 = None
    view_891: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_335, [1, 128, 128]);  addmm_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_894: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_891, [1, 128, 4, 32]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_448: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_894, [0, 2, 1, 3]);  view_894 = None
    
    # No stacktrace found for following nodes
    clone_default_5: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default_3, clone_default_4, clone_default_5, scale = 0.17677669529663687);  clone_default_3 = clone_default_4 = clone_default_5 = None
    getitem_3: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default_1[0];  _scaled_dot_product_flash_attention_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_450: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_3, [0, 2, 1, 3]);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_901: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_450, [1, 128, 128]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_902: "f32[128, 128]" = torch.ops.aten.reshape.default(view_901, [128, 128]);  view_901 = None
    permute_451: "f32[128, 128]" = torch.ops.aten.permute.default(arg1061_1, [1, 0]);  arg1061_1 = None
    addmm_336: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1062_1, view_902, permute_451);  arg1062_1 = view_902 = permute_451 = None
    view_903: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_336, [1, 128, 128]);  addmm_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_882: "f32[128, 512]" = torch.ops.aten.reshape.default(add_332, [128, 512])
    permute_441: "f32[512, 128]" = torch.ops.aten.permute.default(arg1051_1, [1, 0]);  arg1051_1 = None
    addmm_331: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1052_1, view_882, permute_441);  arg1052_1 = view_882 = permute_441 = None
    view_883: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_331, [1, 128, 128]);  addmm_331 = None
    
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
    permute_452: "f32[128, 512]" = torch.ops.aten.permute.default(arg1063_1, [1, 0]);  arg1063_1 = None
    addmm_337: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1064_1, view_904, permute_452);  arg1064_1 = view_904 = permute_452 = None
    view_905: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_337, [1, 128, 512]);  addmm_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_88: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_905);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_906: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_88, [128, 512]);  relu_88 = None
    permute_453: "f32[512, 128]" = torch.ops.aten.permute.default(arg1065_1, [1, 0]);  arg1065_1 = None
    addmm_338: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1066_1, view_906, permute_453);  arg1066_1 = view_906 = permute_453 = None
    view_907: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_338, [1, 128, 128]);  addmm_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_338: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_907, add_337);  view_907 = add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_181: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_338, arg360_1);  add_338 = arg360_1 = None
    add_339: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_181, arg361_1);  mul_181 = arg361_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_908: "f32[128, 128]" = torch.ops.aten.reshape.default(add_339, [128, 128])
    permute_454: "f32[128, 512]" = torch.ops.aten.permute.default(arg1067_1, [1, 0]);  arg1067_1 = None
    addmm_339: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1068_1, view_908, permute_454);  arg1068_1 = view_908 = permute_454 = None
    view_909: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_339, [1, 128, 512]);  addmm_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_89: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_909);  view_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_910: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_89, [128, 512]);  relu_89 = None
    permute_455: "f32[512, 128]" = torch.ops.aten.permute.default(arg1069_1, [1, 0]);  arg1069_1 = None
    addmm_340: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1070_1, view_910, permute_455);  arg1070_1 = view_910 = permute_455 = None
    view_911: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_340, [1, 128, 128]);  addmm_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_340: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_911, add_339);  view_911 = add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_182: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_340, arg362_1);  add_340 = arg362_1 = None
    add_341: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_182, arg363_1);  mul_182 = arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_912: "f32[128, 128]" = torch.ops.aten.reshape.default(add_341, [128, 128])
    permute_456: "f32[128, 512]" = torch.ops.aten.permute.default(arg1071_1, [1, 0]);  arg1071_1 = None
    addmm_341: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1072_1, view_912, permute_456);  arg1072_1 = view_912 = permute_456 = None
    view_913: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_341, [1, 128, 512]);  addmm_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_90: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_913);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_914: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_90, [128, 512]);  relu_90 = None
    permute_457: "f32[512, 128]" = torch.ops.aten.permute.default(arg1073_1, [1, 0]);  arg1073_1 = None
    addmm_342: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1074_1, view_914, permute_457);  arg1074_1 = view_914 = permute_457 = None
    view_915: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_342, [1, 128, 128]);  addmm_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_342: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_915, add_341);  view_915 = add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_183: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_342, arg364_1);  add_342 = arg364_1 = None
    add_343: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_183, arg365_1);  mul_183 = arg365_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_916: "f32[128, 128]" = torch.ops.aten.reshape.default(add_343, [128, 128])
    permute_458: "f32[128, 512]" = torch.ops.aten.permute.default(arg1075_1, [1, 0]);  arg1075_1 = None
    addmm_343: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1076_1, view_916, permute_458);  arg1076_1 = view_916 = permute_458 = None
    view_917: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_343, [1, 128, 512]);  addmm_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_91: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_917);  view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_918: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_91, [128, 512]);  relu_91 = None
    permute_459: "f32[512, 128]" = torch.ops.aten.permute.default(arg1077_1, [1, 0]);  arg1077_1 = None
    addmm_344: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1078_1, view_918, permute_459);  arg1078_1 = view_918 = permute_459 = None
    view_919: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_344, [1, 128, 128]);  addmm_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_344: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_919, add_343);  view_919 = add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_184: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_344, arg366_1);  add_344 = arg366_1 = None
    add_345: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_184, arg367_1);  mul_184 = arg367_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_920: "f32[128, 128]" = torch.ops.aten.reshape.default(add_345, [128, 128]);  add_345 = None
    permute_460: "f32[128, 512]" = torch.ops.aten.permute.default(arg1079_1, [1, 0]);  arg1079_1 = None
    addmm_345: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1080_1, view_920, permute_460);  arg1080_1 = view_920 = permute_460 = None
    view_921: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_345, [1, 128, 512]);  addmm_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_346: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_921, add_332);  view_921 = add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_185: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_346, arg368_1);  add_346 = arg368_1 = None
    add_347: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_185, arg369_1);  mul_185 = arg369_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_924: "f32[128, 512]" = torch.ops.aten.reshape.default(add_347, [128, 512])
    permute_462: "f32[512, 128]" = torch.ops.aten.permute.default(arg1083_1, [1, 0]);  arg1083_1 = None
    addmm_347: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1084_1, view_924, permute_462);  arg1084_1 = view_924 = permute_462 = None
    view_925: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_347, [1, 128, 128]);  addmm_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_187: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_925, arg372_1);  view_925 = arg372_1 = None
    add_349: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_187, arg373_1);  mul_187 = arg373_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_926: "f32[128, 128]" = torch.ops.aten.reshape.default(add_349, [128, 128])
    permute_463: "f32[128, 128]" = torch.ops.aten.permute.default(arg1085_1, [1, 0]);  arg1085_1 = None
    addmm_348: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1086_1, view_926, permute_463);  arg1086_1 = view_926 = permute_463 = None
    view_927: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_348, [1, 128, 128]);  addmm_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_932: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_927, [1, 128, 4, 32]);  view_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_466: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_932, [0, 2, 1, 3]);  view_932 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    view_928: "f32[128, 128]" = torch.ops.aten.reshape.default(add_349, [128, 128]);  add_349 = None
    permute_464: "f32[128, 128]" = torch.ops.aten.permute.default(arg1087_1, [1, 0]);  arg1087_1 = None
    addmm_349: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1088_1, view_928, permute_464);  arg1088_1 = view_928 = permute_464 = None
    view_929: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_349, [1, 128, 128]);  addmm_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_933: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_929, [1, 128, 4, 32]);  view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_467: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_933, [0, 2, 1, 3]);  view_933 = None
    
    # No stacktrace found for following nodes
    clone_default_1: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    view_930: "f32[128, 512]" = torch.ops.aten.reshape.default(add_347, [128, 512])
    permute_465: "f32[512, 128]" = torch.ops.aten.permute.default(arg1089_1, [1, 0]);  arg1089_1 = None
    addmm_350: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1090_1, view_930, permute_465);  arg1090_1 = view_930 = permute_465 = None
    view_931: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_350, [1, 128, 128]);  addmm_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_934: "f32[1, 128, 4, 32]" = torch.ops.aten.reshape.default(view_931, [1, 128, 4, 32]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_468: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None
    
    # No stacktrace found for following nodes
    clone_default_2: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(clone_default, clone_default_1, clone_default_2, scale = 0.17677669529663687);  clone_default = clone_default_1 = clone_default_2 = None
    getitem_2: "f32[1, 4, 128, 32]" = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_470: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_941: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(permute_470, [1, 128, 128]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_942: "f32[128, 128]" = torch.ops.aten.reshape.default(view_941, [128, 128]);  view_941 = None
    permute_471: "f32[128, 128]" = torch.ops.aten.permute.default(arg1091_1, [1, 0]);  arg1091_1 = None
    addmm_351: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1092_1, view_942, permute_471);  arg1092_1 = view_942 = permute_471 = None
    view_943: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_351, [1, 128, 128]);  addmm_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_922: "f32[128, 512]" = torch.ops.aten.reshape.default(add_347, [128, 512])
    permute_461: "f32[512, 128]" = torch.ops.aten.permute.default(arg1081_1, [1, 0]);  arg1081_1 = None
    addmm_346: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1082_1, view_922, permute_461);  arg1082_1 = view_922 = permute_461 = None
    view_923: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_346, [1, 128, 128]);  addmm_346 = None
    
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
    permute_472: "f32[128, 512]" = torch.ops.aten.permute.default(arg1093_1, [1, 0]);  arg1093_1 = None
    addmm_352: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1094_1, view_944, permute_472);  arg1094_1 = view_944 = permute_472 = None
    view_945: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_352, [1, 128, 512]);  addmm_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_92: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_945);  view_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_946: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_92, [128, 512]);  relu_92 = None
    permute_473: "f32[512, 128]" = torch.ops.aten.permute.default(arg1095_1, [1, 0]);  arg1095_1 = None
    addmm_353: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1096_1, view_946, permute_473);  arg1096_1 = view_946 = permute_473 = None
    view_947: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_353, [1, 128, 128]);  addmm_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_353: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_947, add_352);  view_947 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_189: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_353, arg376_1);  add_353 = arg376_1 = None
    add_354: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, arg377_1);  mul_189 = arg377_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_948: "f32[128, 128]" = torch.ops.aten.reshape.default(add_354, [128, 128])
    permute_474: "f32[128, 512]" = torch.ops.aten.permute.default(arg1097_1, [1, 0]);  arg1097_1 = None
    addmm_354: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1098_1, view_948, permute_474);  arg1098_1 = view_948 = permute_474 = None
    view_949: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_354, [1, 128, 512]);  addmm_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_93: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_949);  view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_950: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_93, [128, 512]);  relu_93 = None
    permute_475: "f32[512, 128]" = torch.ops.aten.permute.default(arg1099_1, [1, 0]);  arg1099_1 = None
    addmm_355: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1100_1, view_950, permute_475);  arg1100_1 = view_950 = permute_475 = None
    view_951: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_355, [1, 128, 128]);  addmm_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_355: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_951, add_354);  view_951 = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_190: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_355, arg378_1);  add_355 = arg378_1 = None
    add_356: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_190, arg379_1);  mul_190 = arg379_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_952: "f32[128, 128]" = torch.ops.aten.reshape.default(add_356, [128, 128])
    permute_476: "f32[128, 512]" = torch.ops.aten.permute.default(arg1101_1, [1, 0]);  arg1101_1 = None
    addmm_356: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1102_1, view_952, permute_476);  arg1102_1 = view_952 = permute_476 = None
    view_953: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_356, [1, 128, 512]);  addmm_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_94: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_953);  view_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_954: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_94, [128, 512]);  relu_94 = None
    permute_477: "f32[512, 128]" = torch.ops.aten.permute.default(arg1103_1, [1, 0]);  arg1103_1 = None
    addmm_357: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1104_1, view_954, permute_477);  arg1104_1 = view_954 = permute_477 = None
    view_955: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_357, [1, 128, 128]);  addmm_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_357: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_955, add_356);  view_955 = add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_191: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_357, arg380_1);  add_357 = arg380_1 = None
    add_358: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_191, arg381_1);  mul_191 = arg381_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_956: "f32[128, 128]" = torch.ops.aten.reshape.default(add_358, [128, 128])
    permute_478: "f32[128, 512]" = torch.ops.aten.permute.default(arg1105_1, [1, 0]);  arg1105_1 = None
    addmm_358: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1106_1, view_956, permute_478);  arg1106_1 = view_956 = permute_478 = None
    view_957: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_358, [1, 128, 512]);  addmm_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_95: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_957);  view_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_958: "f32[128, 512]" = torch.ops.aten.reshape.default(relu_95, [128, 512]);  relu_95 = None
    permute_479: "f32[512, 128]" = torch.ops.aten.permute.default(arg1107_1, [1, 0]);  arg1107_1 = None
    addmm_359: "f32[128, 128]" = torch.ops.aten.addmm.default(arg1108_1, view_958, permute_479);  arg1108_1 = view_958 = permute_479 = None
    view_959: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(addmm_359, [1, 128, 128]);  addmm_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_359: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_959, add_358);  view_959 = add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_192: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_359, arg382_1);  add_359 = arg382_1 = None
    add_360: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_192, arg383_1);  mul_192 = arg383_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_960: "f32[128, 128]" = torch.ops.aten.reshape.default(add_360, [128, 128]);  add_360 = None
    permute_480: "f32[128, 512]" = torch.ops.aten.permute.default(arg1109_1, [1, 0]);  arg1109_1 = None
    addmm_360: "f32[128, 512]" = torch.ops.aten.addmm.default(arg1110_1, view_960, permute_480);  arg1110_1 = view_960 = permute_480 = None
    view_961: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_360, [1, 128, 512]);  addmm_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_361: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_961, add_347);  view_961 = add_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_193: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_361, arg384_1);  add_361 = arg384_1 = None
    add_362: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_193, arg385_1);  mul_193 = arg385_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1404, code: logits = self.qa_outputs(sequence_output)
    view_962: "f32[128, 512]" = torch.ops.aten.reshape.default(add_362, [128, 512]);  add_362 = None
    permute_481: "f32[512, 2]" = torch.ops.aten.permute.default(arg1111_1, [1, 0]);  arg1111_1 = None
    addmm_361: "f32[128, 2]" = torch.ops.aten.addmm.default(arg1112_1, view_962, permute_481);  arg1112_1 = view_962 = permute_481 = None
    view_963: "f32[1, 128, 2]" = torch.ops.aten.reshape.default(addmm_361, [1, 128, 2]);  addmm_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1405, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_963, [1, 1], 2);  view_963 = None
    getitem: "f32[1, 128, 1]" = split_with_sizes[0]
    getitem_1: "f32[1, 128, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1418, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg1115_1, 0);  arg1115_1 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem, -1);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1406, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_73: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    amax_24: "f32[1, 1]" = torch.ops.aten.amax.default(clone_73, [1], True)
    sub_25: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_73, amax_24);  amax_24 = None
    exp_24: "f32[1, 128]" = torch.ops.aten.exp.default(sub_25)
    sum_25: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_26: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full_default_2);  ne = full_default_2 = None
    unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_26, 1, unsqueeze_2);  sub_26 = unsqueeze_2 = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128);  clamp_max = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1419, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg1116_1, 0);  arg1116_1 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_1, -1);  getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1407, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_74: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    amax_25: "f32[1, 1]" = torch.ops.aten.amax.default(clone_74, [1], True)
    sub_27: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_74, amax_25);  amax_25 = None
    exp_25: "f32[1, 128]" = torch.ops.aten.exp.default(sub_27)
    sum_28: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_28);  sum_28 = None
    sub_28: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_27, log_1);  sub_27 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
    full_default_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_2: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_4);  ne_3 = full_default_4 = None
    unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_28, 1, unsqueeze_3);  sub_28 = unsqueeze_3 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_3: "f32[1]" = torch.ops.aten.where.self(ne_4, neg_1, full_default_5);  ne_4 = neg_1 = full_default_5 = None
    sum_30: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128);  clamp_max_1 = None
    sum_29: "i64[]" = torch.ops.aten.sum.default(ne_5);  ne_5 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_30, convert_element_type_1);  sum_30 = convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1424, code: total_loss = (start_loss + end_loss) / 2
    add_363: "f32[]" = torch.ops.aten.add.Tensor(div_48, div_49);  div_48 = div_49 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(add_363, 2);  add_363 = None
    return (div_50, clone_73, clone_74)
    