from __future__ import annotations



def forward(self, arg0_1: "f32[1026, 1024]", arg1_1: "f32[1026, 1024]", arg2_1: "f32[50265, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024, 1024]", arg14_1: "f32[1024]", arg15_1: "f32[1024]", arg16_1: "f32[1024]", arg17_1: "f32[4096, 1024]", arg18_1: "f32[4096]", arg19_1: "f32[1024, 4096]", arg20_1: "f32[1024]", arg21_1: "f32[1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024, 1024]", arg30_1: "f32[1024]", arg31_1: "f32[1024]", arg32_1: "f32[1024]", arg33_1: "f32[4096, 1024]", arg34_1: "f32[4096]", arg35_1: "f32[1024, 4096]", arg36_1: "f32[1024]", arg37_1: "f32[1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024, 1024]", arg44_1: "f32[1024]", arg45_1: "f32[1024, 1024]", arg46_1: "f32[1024]", arg47_1: "f32[1024]", arg48_1: "f32[1024]", arg49_1: "f32[4096, 1024]", arg50_1: "f32[4096]", arg51_1: "f32[1024, 4096]", arg52_1: "f32[1024]", arg53_1: "f32[1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024, 1024]", arg60_1: "f32[1024]", arg61_1: "f32[1024, 1024]", arg62_1: "f32[1024]", arg63_1: "f32[1024]", arg64_1: "f32[1024]", arg65_1: "f32[4096, 1024]", arg66_1: "f32[4096]", arg67_1: "f32[1024, 4096]", arg68_1: "f32[1024]", arg69_1: "f32[1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024, 1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024, 1024]", arg78_1: "f32[1024]", arg79_1: "f32[1024]", arg80_1: "f32[1024]", arg81_1: "f32[4096, 1024]", arg82_1: "f32[4096]", arg83_1: "f32[1024, 4096]", arg84_1: "f32[1024]", arg85_1: "f32[1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024, 1024]", arg92_1: "f32[1024]", arg93_1: "f32[1024, 1024]", arg94_1: "f32[1024]", arg95_1: "f32[1024]", arg96_1: "f32[1024]", arg97_1: "f32[4096, 1024]", arg98_1: "f32[4096]", arg99_1: "f32[1024, 4096]", arg100_1: "f32[1024]", arg101_1: "f32[1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024, 1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024, 1024]", arg110_1: "f32[1024]", arg111_1: "f32[1024]", arg112_1: "f32[1024]", arg113_1: "f32[4096, 1024]", arg114_1: "f32[4096]", arg115_1: "f32[1024, 4096]", arg116_1: "f32[1024]", arg117_1: "f32[1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024, 1024]", arg124_1: "f32[1024]", arg125_1: "f32[1024, 1024]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[1024]", arg129_1: "f32[4096, 1024]", arg130_1: "f32[4096]", arg131_1: "f32[1024, 4096]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024, 1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024, 1024]", arg142_1: "f32[1024]", arg143_1: "f32[1024]", arg144_1: "f32[1024]", arg145_1: "f32[4096, 1024]", arg146_1: "f32[4096]", arg147_1: "f32[1024, 4096]", arg148_1: "f32[1024]", arg149_1: "f32[1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024, 1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024, 1024]", arg158_1: "f32[1024]", arg159_1: "f32[1024]", arg160_1: "f32[1024]", arg161_1: "f32[4096, 1024]", arg162_1: "f32[4096]", arg163_1: "f32[1024, 4096]", arg164_1: "f32[1024]", arg165_1: "f32[1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024, 1024]", arg172_1: "f32[1024]", arg173_1: "f32[1024, 1024]", arg174_1: "f32[1024]", arg175_1: "f32[1024]", arg176_1: "f32[1024]", arg177_1: "f32[4096, 1024]", arg178_1: "f32[4096]", arg179_1: "f32[1024, 4096]", arg180_1: "f32[1024]", arg181_1: "f32[1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024, 1024]", arg188_1: "f32[1024]", arg189_1: "f32[1024, 1024]", arg190_1: "f32[1024]", arg191_1: "f32[1024]", arg192_1: "f32[1024]", arg193_1: "f32[4096, 1024]", arg194_1: "f32[4096]", arg195_1: "f32[1024, 4096]", arg196_1: "f32[1024]", arg197_1: "f32[1024]", arg198_1: "f32[1024]", arg199_1: "f32[50265, 1024]", arg200_1: "f32[1024]", arg201_1: "f32[1024]", arg202_1: "f32[1024]", arg203_1: "f32[1024]", arg204_1: "f32[1024, 1024]", arg205_1: "f32[1024]", arg206_1: "f32[1024, 1024]", arg207_1: "f32[1024]", arg208_1: "f32[1024, 1024]", arg209_1: "f32[1024]", arg210_1: "f32[1024, 1024]", arg211_1: "f32[1024]", arg212_1: "f32[1024]", arg213_1: "f32[1024]", arg214_1: "f32[1024, 1024]", arg215_1: "f32[1024]", arg216_1: "f32[1024, 1024]", arg217_1: "f32[1024]", arg218_1: "f32[1024, 1024]", arg219_1: "f32[1024]", arg220_1: "f32[1024, 1024]", arg221_1: "f32[1024]", arg222_1: "f32[1024]", arg223_1: "f32[1024]", arg224_1: "f32[4096, 1024]", arg225_1: "f32[4096]", arg226_1: "f32[1024, 4096]", arg227_1: "f32[1024]", arg228_1: "f32[1024]", arg229_1: "f32[1024]", arg230_1: "f32[1024, 1024]", arg231_1: "f32[1024]", arg232_1: "f32[1024, 1024]", arg233_1: "f32[1024]", arg234_1: "f32[1024, 1024]", arg235_1: "f32[1024]", arg236_1: "f32[1024, 1024]", arg237_1: "f32[1024]", arg238_1: "f32[1024]", arg239_1: "f32[1024]", arg240_1: "f32[1024, 1024]", arg241_1: "f32[1024]", arg242_1: "f32[1024, 1024]", arg243_1: "f32[1024]", arg244_1: "f32[1024, 1024]", arg245_1: "f32[1024]", arg246_1: "f32[1024, 1024]", arg247_1: "f32[1024]", arg248_1: "f32[1024]", arg249_1: "f32[1024]", arg250_1: "f32[4096, 1024]", arg251_1: "f32[4096]", arg252_1: "f32[1024, 4096]", arg253_1: "f32[1024]", arg254_1: "f32[1024]", arg255_1: "f32[1024]", arg256_1: "f32[1024, 1024]", arg257_1: "f32[1024]", arg258_1: "f32[1024, 1024]", arg259_1: "f32[1024]", arg260_1: "f32[1024, 1024]", arg261_1: "f32[1024]", arg262_1: "f32[1024, 1024]", arg263_1: "f32[1024]", arg264_1: "f32[1024]", arg265_1: "f32[1024]", arg266_1: "f32[1024, 1024]", arg267_1: "f32[1024]", arg268_1: "f32[1024, 1024]", arg269_1: "f32[1024]", arg270_1: "f32[1024, 1024]", arg271_1: "f32[1024]", arg272_1: "f32[1024, 1024]", arg273_1: "f32[1024]", arg274_1: "f32[1024]", arg275_1: "f32[1024]", arg276_1: "f32[4096, 1024]", arg277_1: "f32[4096]", arg278_1: "f32[1024, 4096]", arg279_1: "f32[1024]", arg280_1: "f32[1024]", arg281_1: "f32[1024]", arg282_1: "f32[1024, 1024]", arg283_1: "f32[1024]", arg284_1: "f32[1024, 1024]", arg285_1: "f32[1024]", arg286_1: "f32[1024, 1024]", arg287_1: "f32[1024]", arg288_1: "f32[1024, 1024]", arg289_1: "f32[1024]", arg290_1: "f32[1024]", arg291_1: "f32[1024]", arg292_1: "f32[1024, 1024]", arg293_1: "f32[1024]", arg294_1: "f32[1024, 1024]", arg295_1: "f32[1024]", arg296_1: "f32[1024, 1024]", arg297_1: "f32[1024]", arg298_1: "f32[1024, 1024]", arg299_1: "f32[1024]", arg300_1: "f32[1024]", arg301_1: "f32[1024]", arg302_1: "f32[4096, 1024]", arg303_1: "f32[4096]", arg304_1: "f32[1024, 4096]", arg305_1: "f32[1024]", arg306_1: "f32[1024]", arg307_1: "f32[1024]", arg308_1: "f32[1024, 1024]", arg309_1: "f32[1024]", arg310_1: "f32[1024, 1024]", arg311_1: "f32[1024]", arg312_1: "f32[1024, 1024]", arg313_1: "f32[1024]", arg314_1: "f32[1024, 1024]", arg315_1: "f32[1024]", arg316_1: "f32[1024]", arg317_1: "f32[1024]", arg318_1: "f32[1024, 1024]", arg319_1: "f32[1024]", arg320_1: "f32[1024, 1024]", arg321_1: "f32[1024]", arg322_1: "f32[1024, 1024]", arg323_1: "f32[1024]", arg324_1: "f32[1024, 1024]", arg325_1: "f32[1024]", arg326_1: "f32[1024]", arg327_1: "f32[1024]", arg328_1: "f32[4096, 1024]", arg329_1: "f32[4096]", arg330_1: "f32[1024, 4096]", arg331_1: "f32[1024]", arg332_1: "f32[1024]", arg333_1: "f32[1024]", arg334_1: "f32[1024, 1024]", arg335_1: "f32[1024]", arg336_1: "f32[1024, 1024]", arg337_1: "f32[1024]", arg338_1: "f32[1024, 1024]", arg339_1: "f32[1024]", arg340_1: "f32[1024, 1024]", arg341_1: "f32[1024]", arg342_1: "f32[1024]", arg343_1: "f32[1024]", arg344_1: "f32[1024, 1024]", arg345_1: "f32[1024]", arg346_1: "f32[1024, 1024]", arg347_1: "f32[1024]", arg348_1: "f32[1024, 1024]", arg349_1: "f32[1024]", arg350_1: "f32[1024, 1024]", arg351_1: "f32[1024]", arg352_1: "f32[1024]", arg353_1: "f32[1024]", arg354_1: "f32[4096, 1024]", arg355_1: "f32[4096]", arg356_1: "f32[1024, 4096]", arg357_1: "f32[1024]", arg358_1: "f32[1024]", arg359_1: "f32[1024]", arg360_1: "f32[1024, 1024]", arg361_1: "f32[1024]", arg362_1: "f32[1024, 1024]", arg363_1: "f32[1024]", arg364_1: "f32[1024, 1024]", arg365_1: "f32[1024]", arg366_1: "f32[1024, 1024]", arg367_1: "f32[1024]", arg368_1: "f32[1024]", arg369_1: "f32[1024]", arg370_1: "f32[1024, 1024]", arg371_1: "f32[1024]", arg372_1: "f32[1024, 1024]", arg373_1: "f32[1024]", arg374_1: "f32[1024, 1024]", arg375_1: "f32[1024]", arg376_1: "f32[1024, 1024]", arg377_1: "f32[1024]", arg378_1: "f32[1024]", arg379_1: "f32[1024]", arg380_1: "f32[4096, 1024]", arg381_1: "f32[4096]", arg382_1: "f32[1024, 4096]", arg383_1: "f32[1024]", arg384_1: "f32[1024]", arg385_1: "f32[1024]", arg386_1: "f32[1024, 1024]", arg387_1: "f32[1024]", arg388_1: "f32[1024, 1024]", arg389_1: "f32[1024]", arg390_1: "f32[1024, 1024]", arg391_1: "f32[1024]", arg392_1: "f32[1024, 1024]", arg393_1: "f32[1024]", arg394_1: "f32[1024]", arg395_1: "f32[1024]", arg396_1: "f32[1024, 1024]", arg397_1: "f32[1024]", arg398_1: "f32[1024, 1024]", arg399_1: "f32[1024]", arg400_1: "f32[1024, 1024]", arg401_1: "f32[1024]", arg402_1: "f32[1024, 1024]", arg403_1: "f32[1024]", arg404_1: "f32[1024]", arg405_1: "f32[1024]", arg406_1: "f32[4096, 1024]", arg407_1: "f32[4096]", arg408_1: "f32[1024, 4096]", arg409_1: "f32[1024]", arg410_1: "f32[1024]", arg411_1: "f32[1024]", arg412_1: "f32[1024, 1024]", arg413_1: "f32[1024]", arg414_1: "f32[1024, 1024]", arg415_1: "f32[1024]", arg416_1: "f32[1024, 1024]", arg417_1: "f32[1024]", arg418_1: "f32[1024, 1024]", arg419_1: "f32[1024]", arg420_1: "f32[1024]", arg421_1: "f32[1024]", arg422_1: "f32[1024, 1024]", arg423_1: "f32[1024]", arg424_1: "f32[1024, 1024]", arg425_1: "f32[1024]", arg426_1: "f32[1024, 1024]", arg427_1: "f32[1024]", arg428_1: "f32[1024, 1024]", arg429_1: "f32[1024]", arg430_1: "f32[1024]", arg431_1: "f32[1024]", arg432_1: "f32[4096, 1024]", arg433_1: "f32[4096]", arg434_1: "f32[1024, 4096]", arg435_1: "f32[1024]", arg436_1: "f32[1024]", arg437_1: "f32[1024]", arg438_1: "f32[1024, 1024]", arg439_1: "f32[1024]", arg440_1: "f32[1024, 1024]", arg441_1: "f32[1024]", arg442_1: "f32[1024, 1024]", arg443_1: "f32[1024]", arg444_1: "f32[1024, 1024]", arg445_1: "f32[1024]", arg446_1: "f32[1024]", arg447_1: "f32[1024]", arg448_1: "f32[1024, 1024]", arg449_1: "f32[1024]", arg450_1: "f32[1024, 1024]", arg451_1: "f32[1024]", arg452_1: "f32[1024, 1024]", arg453_1: "f32[1024]", arg454_1: "f32[1024, 1024]", arg455_1: "f32[1024]", arg456_1: "f32[1024]", arg457_1: "f32[1024]", arg458_1: "f32[4096, 1024]", arg459_1: "f32[4096]", arg460_1: "f32[1024, 4096]", arg461_1: "f32[1024]", arg462_1: "f32[1024]", arg463_1: "f32[1024]", arg464_1: "f32[1024, 1024]", arg465_1: "f32[1024]", arg466_1: "f32[1024, 1024]", arg467_1: "f32[1024]", arg468_1: "f32[1024, 1024]", arg469_1: "f32[1024]", arg470_1: "f32[1024, 1024]", arg471_1: "f32[1024]", arg472_1: "f32[1024]", arg473_1: "f32[1024]", arg474_1: "f32[1024, 1024]", arg475_1: "f32[1024]", arg476_1: "f32[1024, 1024]", arg477_1: "f32[1024]", arg478_1: "f32[1024, 1024]", arg479_1: "f32[1024]", arg480_1: "f32[1024, 1024]", arg481_1: "f32[1024]", arg482_1: "f32[1024]", arg483_1: "f32[1024]", arg484_1: "f32[4096, 1024]", arg485_1: "f32[4096]", arg486_1: "f32[1024, 4096]", arg487_1: "f32[1024]", arg488_1: "f32[1024]", arg489_1: "f32[1024]", arg490_1: "f32[1024, 1024]", arg491_1: "f32[1024]", arg492_1: "f32[1024, 1024]", arg493_1: "f32[1024]", arg494_1: "f32[1024, 1024]", arg495_1: "f32[1024]", arg496_1: "f32[1024, 1024]", arg497_1: "f32[1024]", arg498_1: "f32[1024]", arg499_1: "f32[1024]", arg500_1: "f32[1024, 1024]", arg501_1: "f32[1024]", arg502_1: "f32[1024, 1024]", arg503_1: "f32[1024]", arg504_1: "f32[1024, 1024]", arg505_1: "f32[1024]", arg506_1: "f32[1024, 1024]", arg507_1: "f32[1024]", arg508_1: "f32[1024]", arg509_1: "f32[1024]", arg510_1: "f32[4096, 1024]", arg511_1: "f32[4096]", arg512_1: "f32[1024, 4096]", arg513_1: "f32[1024]", arg514_1: "f32[1024]", arg515_1: "f32[1024]", arg516_1: "f32[50265, 1024]", arg517_1: "f32[1, 50265]", arg518_1: "i64[1, 1024]", arg519_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:787, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.reshape.default(arg519_1, [-1, 1024]);  arg519_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:794, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg2_1, view, 1);  arg2_1 = view = None
    mul: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:130, code: positions = torch.arange(
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:132, code: ).expand(bsz, -1)
    expand: "i64[1, 1024]" = torch.ops.aten.expand.default(iota, [1, -1]);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:134, code: return super().forward(positions + self.offset)
    add: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg0_1, add);  arg0_1 = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:798, code: hidden_states = inputs_embeds + embed_pos.to(inputs_embeds.device)
    add_1: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:799, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    sub_1: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    add_2: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    mul_1: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_2: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_3: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  getitem_3 = None
    add_4: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_3: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_4: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
    add_5: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, arg6_1);  mul_4 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_5, [1024, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg8_1, view_1, permute);  arg8_1 = view_1 = permute = None
    view_2: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 1024]);  addmm = None
    mul_5: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_5, [1, 1024, 16, 64]);  mul_5 = None
    permute_5: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_5: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_10: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_5, [16, -1, 64]);  clone_5 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_69: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_5, [1024, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_1: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_3, permute_1);  arg10_1 = view_3 = permute_1 = None
    view_4: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_1, [1, 1024, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_4, [1, -1, 16, 64]);  view_4 = None
    permute_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    clone_3: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_11: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_3, [16, -1, 64]);  clone_3 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_70: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_5, [1024, 1024]);  add_5 = None
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    addmm_2: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg12_1, view_6, permute_3);  arg12_1 = view_6 = permute_3 = None
    view_7: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_2, [1, 1024, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_7, [1, -1, 16, 64]);  view_7 = None
    permute_4: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_4: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_12: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_4, [16, -1, 64]);  clone_4 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_71: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
    _scaled_dot_product_flash_attention_default_23 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_69, unsqueeze_default_70, unsqueeze_default_71, scale = 1.0);  unsqueeze_default_69 = unsqueeze_default_70 = unsqueeze_default_71 = None
    getitem_151: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_23[0];  _scaled_dot_product_flash_attention_default_23 = None
    squeeze_dim_23: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_151, 0);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_13: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_23, [1, 16, 1024, 64]);  squeeze_dim_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_7: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_14: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_7, [1, 1024, 1024]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_15: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_14, [1024, 1024]);  view_14 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    addmm_3: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg14_1, view_15, permute_8);  arg14_1 = view_15 = permute_8 = None
    view_16: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_3, [1, 1024, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_6: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_3, view_16);  add_3 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_4: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_5);  getitem_5 = None
    add_7: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    mul_6: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_7: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_6, arg15_1);  mul_6 = arg15_1 = None
    add_8: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_7, arg16_1);  mul_7 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_8, [1024, 1024]);  add_8 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    addmm_4: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg18_1, view_17, permute_9);  arg18_1 = view_17 = permute_9 = None
    view_18: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_4, [1, 1024, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_8: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_9: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_8, add_9);  mul_8 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_19: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_10, [1024, 4096]);  mul_10 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_5: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg20_1, view_19, permute_10);  arg20_1 = view_19 = permute_10 = None
    view_20: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_5, [1, 1024, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_6, view_20);  add_6 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  getitem_7 = None
    add_11: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_11: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_12: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None
    add_12: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_12, arg22_1);  mul_12 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_21: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_12, [1024, 1024])
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_6: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_21, permute_11);  arg24_1 = view_21 = permute_11 = None
    view_22: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_6, [1, 1024, 1024]);  addmm_6 = None
    mul_13: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_22, 0.125);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_13, [1, 1024, 16, 64]);  mul_13 = None
    permute_16: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_13: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_30: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_13, [16, -1, 64]);  clone_13 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_66: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_23: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_12, [1024, 1024])
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_7: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_23, permute_12);  arg26_1 = view_23 = permute_12 = None
    view_24: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_7, [1, 1024, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_24, [1, -1, 16, 64]);  view_24 = None
    permute_13: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_11: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_31: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_11, [16, -1, 64]);  clone_11 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_67: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_26: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_12, [1024, 1024]);  add_12 = None
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_8: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg28_1, view_26, permute_14);  arg28_1 = view_26 = permute_14 = None
    view_27: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_8, [1, 1024, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_27, [1, -1, 16, 64]);  view_27 = None
    permute_15: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_12: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_32: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_12, [16, -1, 64]);  clone_12 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_68: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
    _scaled_dot_product_flash_attention_default_22 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_66, unsqueeze_default_67, unsqueeze_default_68, scale = 1.0);  unsqueeze_default_66 = unsqueeze_default_67 = unsqueeze_default_68 = None
    getitem_150: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_22[0];  _scaled_dot_product_flash_attention_default_22 = None
    squeeze_dim_22: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_150, 0);  getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_33: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_22, [1, 16, 1024, 64]);  squeeze_dim_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_15: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_34: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_15, [1, 1024, 1024]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_35: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_34, [1024, 1024]);  view_34 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    addmm_9: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg30_1, view_35, permute_19);  arg30_1 = view_35 = permute_19 = None
    view_36: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_9, [1, 1024, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_13: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_10, view_36);  add_10 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_7: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_13, getitem_9);  getitem_9 = None
    add_14: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_14: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
    mul_15: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_14, arg31_1);  mul_14 = arg31_1 = None
    add_15: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_15, arg32_1);  mul_15 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_37: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_15, [1024, 1024]);  add_15 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    addmm_10: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg34_1, view_37, permute_20);  arg34_1 = view_37 = permute_20 = None
    view_38: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_10, [1, 1024, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_16: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_17: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_16: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_18: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_16, add_16);  mul_16 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_39: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_18, [1024, 4096]);  mul_18 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    addmm_11: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg36_1, view_39, permute_21);  arg36_1 = view_39 = permute_21 = None
    view_40: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_11, [1, 1024, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_17: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_13, view_40);  add_13 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  getitem_11 = None
    add_18: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_19: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_20: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_19, arg37_1);  mul_19 = arg37_1 = None
    add_19: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_20, arg38_1);  mul_20 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_41: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_19, [1024, 1024])
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_12: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg40_1, view_41, permute_22);  arg40_1 = view_41 = permute_22 = None
    view_42: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_12, [1, 1024, 1024]);  addmm_12 = None
    mul_21: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_42, 0.125);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_21, [1, 1024, 16, 64]);  mul_21 = None
    permute_27: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_21: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_50: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_21, [16, -1, 64]);  clone_21 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_63: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_43: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_19, [1024, 1024])
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_13: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_43, permute_23);  arg42_1 = view_43 = permute_23 = None
    view_44: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_13, [1, 1024, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_44, [1, -1, 16, 64]);  view_44 = None
    permute_24: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_19: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_51: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_19, [16, -1, 64]);  clone_19 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_64: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_51, 0);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_46: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_19, [1024, 1024]);  add_19 = None
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    addmm_14: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg44_1, view_46, permute_25);  arg44_1 = view_46 = permute_25 = None
    view_47: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_14, [1, 1024, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_47, [1, -1, 16, 64]);  view_47 = None
    permute_26: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_20: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_52: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_20, [16, -1, 64]);  clone_20 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_65: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_52, 0);  view_52 = None
    _scaled_dot_product_flash_attention_default_21 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_63, unsqueeze_default_64, unsqueeze_default_65, scale = 1.0);  unsqueeze_default_63 = unsqueeze_default_64 = unsqueeze_default_65 = None
    getitem_149: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_21[0];  _scaled_dot_product_flash_attention_default_21 = None
    squeeze_dim_21: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_149, 0);  getitem_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_53: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_21, [1, 16, 1024, 64]);  squeeze_dim_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_23: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_54: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_23, [1, 1024, 1024]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_55: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_54, [1024, 1024]);  view_54 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    addmm_15: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg46_1, view_55, permute_30);  arg46_1 = view_55 = permute_30 = None
    view_56: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_15, [1, 1024, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_20: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_17, view_56);  add_17 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_10: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_20, getitem_13);  getitem_13 = None
    add_21: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    mul_22: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
    mul_23: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg47_1);  mul_22 = arg47_1 = None
    add_22: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg48_1);  mul_23 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_57: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_22, [1024, 1024]);  add_22 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    addmm_16: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg50_1, view_57, permute_31);  arg50_1 = view_57 = permute_31 = None
    view_58: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_16, [1, 1024, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_24: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_25: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_2: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_23: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_26: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_24, add_23);  mul_24 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_59: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_26, [1024, 4096]);  mul_26 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    addmm_17: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg52_1, view_59, permute_32);  arg52_1 = view_59 = permute_32 = None
    view_60: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_17, [1, 1024, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_24: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_20, view_60);  add_20 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  getitem_15 = None
    add_25: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_27: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_28: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_27, arg53_1);  mul_27 = arg53_1 = None
    add_26: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_28, arg54_1);  mul_28 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_61: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_26, [1024, 1024])
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_18: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg56_1, view_61, permute_33);  arg56_1 = view_61 = permute_33 = None
    view_62: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_18, [1, 1024, 1024]);  addmm_18 = None
    mul_29: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_62, 0.125);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_29, [1, 1024, 16, 64]);  mul_29 = None
    permute_38: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_29: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_70: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_29, [16, -1, 64]);  clone_29 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_60: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_70, 0);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_63: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_26, [1024, 1024])
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_19: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg58_1, view_63, permute_34);  arg58_1 = view_63 = permute_34 = None
    view_64: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_19, [1, 1024, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_64, [1, -1, 16, 64]);  view_64 = None
    permute_35: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_27: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_71: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_27, [16, -1, 64]);  clone_27 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_61: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_66: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_26, [1024, 1024]);  add_26 = None
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    addmm_20: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg60_1, view_66, permute_36);  arg60_1 = view_66 = permute_36 = None
    view_67: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_20, [1, 1024, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_67, [1, -1, 16, 64]);  view_67 = None
    permute_37: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_28: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_72: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_28, [16, -1, 64]);  clone_28 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_62: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
    _scaled_dot_product_flash_attention_default_20 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_60, unsqueeze_default_61, unsqueeze_default_62, scale = 1.0);  unsqueeze_default_60 = unsqueeze_default_61 = unsqueeze_default_62 = None
    getitem_148: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_20[0];  _scaled_dot_product_flash_attention_default_20 = None
    squeeze_dim_20: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_148, 0);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_73: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_20, [1, 16, 1024, 64]);  squeeze_dim_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_31: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_74: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_31, [1, 1024, 1024]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_75: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_74, [1024, 1024]);  view_74 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    addmm_21: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg62_1, view_75, permute_41);  arg62_1 = view_75 = permute_41 = None
    view_76: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_21, [1, 1024, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_27: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_24, view_76);  add_24 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_13: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_27, getitem_17);  getitem_17 = None
    add_28: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    mul_30: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
    mul_31: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_30, arg63_1);  mul_30 = arg63_1 = None
    add_29: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_31, arg64_1);  mul_31 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_77: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_29, [1024, 1024]);  add_29 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    addmm_22: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg66_1, view_77, permute_42);  arg66_1 = view_77 = permute_42 = None
    view_78: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_22, [1, 1024, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_32: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_33: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_3: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_30: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_32, add_30);  mul_32 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_79: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_34, [1024, 4096]);  mul_34 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    addmm_23: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg68_1, view_79, permute_43);  arg68_1 = view_79 = permute_43 = None
    view_80: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_23, [1, 1024, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_31: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_27, view_80);  add_27 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  getitem_19 = None
    add_32: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_35: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_36: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
    add_33: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_36, arg70_1);  mul_36 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_81: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_33, [1024, 1024])
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_24: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_81, permute_44);  arg72_1 = view_81 = permute_44 = None
    view_82: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_24, [1, 1024, 1024]);  addmm_24 = None
    mul_37: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_82, 0.125);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_37, [1, 1024, 16, 64]);  mul_37 = None
    permute_49: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_37: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_90: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_37, [16, -1, 64]);  clone_37 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_57: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_90, 0);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_83: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_33, [1024, 1024])
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_25: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_83, permute_45);  arg74_1 = view_83 = permute_45 = None
    view_84: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_25, [1, 1024, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_84, [1, -1, 16, 64]);  view_84 = None
    permute_46: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    clone_35: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_91: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_35, [16, -1, 64]);  clone_35 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_58: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_91, 0);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_86: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_33, [1024, 1024]);  add_33 = None
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    addmm_26: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg76_1, view_86, permute_47);  arg76_1 = view_86 = permute_47 = None
    view_87: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_26, [1, 1024, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_87, [1, -1, 16, 64]);  view_87 = None
    permute_48: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
    clone_36: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_92: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_36, [16, -1, 64]);  clone_36 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_59: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_92, 0);  view_92 = None
    _scaled_dot_product_flash_attention_default_19 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_57, unsqueeze_default_58, unsqueeze_default_59, scale = 1.0);  unsqueeze_default_57 = unsqueeze_default_58 = unsqueeze_default_59 = None
    getitem_147: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_19[0];  _scaled_dot_product_flash_attention_default_19 = None
    squeeze_dim_19: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_147, 0);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_93: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_19, [1, 16, 1024, 64]);  squeeze_dim_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_39: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_94: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_39, [1, 1024, 1024]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_95: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_94, [1024, 1024]);  view_94 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_27: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg78_1, view_95, permute_52);  arg78_1 = view_95 = permute_52 = None
    view_96: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_27, [1, 1024, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_34: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_31, view_96);  add_31 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_16: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_34, getitem_21);  getitem_21 = None
    add_35: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    mul_38: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_39: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_38, arg79_1);  mul_38 = arg79_1 = None
    add_36: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_39, arg80_1);  mul_39 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_97: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_36, [1024, 1024]);  add_36 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    addmm_28: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg82_1, view_97, permute_53);  arg82_1 = view_97 = permute_53 = None
    view_98: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_28, [1, 1024, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_41: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_4: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_37: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_42: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_40, add_37);  mul_40 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_99: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_42, [1024, 4096]);  mul_42 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    addmm_29: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg84_1, view_99, permute_54);  arg84_1 = view_99 = permute_54 = None
    view_100: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_29, [1, 1024, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_34, view_100);  add_34 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  getitem_23 = None
    add_39: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_43: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_44: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_43, arg85_1);  mul_43 = arg85_1 = None
    add_40: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_44, arg86_1);  mul_44 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_101: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_40, [1024, 1024])
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_30: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg88_1, view_101, permute_55);  arg88_1 = view_101 = permute_55 = None
    view_102: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_30, [1, 1024, 1024]);  addmm_30 = None
    mul_45: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_102, 0.125);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_45, [1, 1024, 16, 64]);  mul_45 = None
    permute_60: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_45: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_110: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_45, [16, -1, 64]);  clone_45 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_54: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_103: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_40, [1024, 1024])
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_31: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg90_1, view_103, permute_56);  arg90_1 = view_103 = permute_56 = None
    view_104: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_31, [1, 1024, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_104, [1, -1, 16, 64]);  view_104 = None
    permute_57: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    clone_43: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_111: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_43, [16, -1, 64]);  clone_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_55: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_111, 0);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_106: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_40, [1024, 1024]);  add_40 = None
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm_32: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg92_1, view_106, permute_58);  arg92_1 = view_106 = permute_58 = None
    view_107: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_32, [1, 1024, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_108: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_107, [1, -1, 16, 64]);  view_107 = None
    permute_59: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    clone_44: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_112: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_44, [16, -1, 64]);  clone_44 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_56: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_112, 0);  view_112 = None
    _scaled_dot_product_flash_attention_default_18 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_54, unsqueeze_default_55, unsqueeze_default_56, scale = 1.0);  unsqueeze_default_54 = unsqueeze_default_55 = unsqueeze_default_56 = None
    getitem_146: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_18[0];  _scaled_dot_product_flash_attention_default_18 = None
    squeeze_dim_18: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_146, 0);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_113: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_18, [1, 16, 1024, 64]);  squeeze_dim_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_47: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_114: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_47, [1, 1024, 1024]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_115: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_114, [1024, 1024]);  view_114 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_33: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg94_1, view_115, permute_63);  arg94_1 = view_115 = permute_63 = None
    view_116: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_33, [1, 1024, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_41: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_38, view_116);  add_38 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_19: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_25);  getitem_25 = None
    add_42: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_46: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_47: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_46, arg95_1);  mul_46 = arg95_1 = None
    add_43: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_47, arg96_1);  mul_47 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_117: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_43, [1024, 1024]);  add_43 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    addmm_34: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg98_1, view_117, permute_64);  arg98_1 = view_117 = permute_64 = None
    view_118: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_34, [1, 1024, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_49: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_5: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_44: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_50: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_48, add_44);  mul_48 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_119: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_50, [1024, 4096]);  mul_50 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    addmm_35: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg100_1, view_119, permute_65);  arg100_1 = view_119 = permute_65 = None
    view_120: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_35, [1, 1024, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_45: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_41, view_120);  add_41 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_20: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_27);  getitem_27 = None
    add_46: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_51: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_52: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_51, arg101_1);  mul_51 = arg101_1 = None
    add_47: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_52, arg102_1);  mul_52 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_121: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_47, [1024, 1024])
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    addmm_36: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_121, permute_66);  arg104_1 = view_121 = permute_66 = None
    view_122: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_36, [1, 1024, 1024]);  addmm_36 = None
    mul_53: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_122, 0.125);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_129: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_53, [1, 1024, 16, 64]);  mul_53 = None
    permute_71: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    clone_53: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_130: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_53, [16, -1, 64]);  clone_53 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_51: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_130, 0);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_123: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_47, [1024, 1024])
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_37: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg106_1, view_123, permute_67);  arg106_1 = view_123 = permute_67 = None
    view_124: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_37, [1, 1024, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_125: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_124, [1, -1, 16, 64]);  view_124 = None
    permute_68: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
    clone_51: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_131: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_51, [16, -1, 64]);  clone_51 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_52: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_131, 0);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_126: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_47, [1024, 1024]);  add_47 = None
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm_38: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg108_1, view_126, permute_69);  arg108_1 = view_126 = permute_69 = None
    view_127: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_38, [1, 1024, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_128: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_127, [1, -1, 16, 64]);  view_127 = None
    permute_70: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    clone_52: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_132: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_52, [16, -1, 64]);  clone_52 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_53: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_132, 0);  view_132 = None
    _scaled_dot_product_flash_attention_default_17 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_51, unsqueeze_default_52, unsqueeze_default_53, scale = 1.0);  unsqueeze_default_51 = unsqueeze_default_52 = unsqueeze_default_53 = None
    getitem_145: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_17[0];  _scaled_dot_product_flash_attention_default_17 = None
    squeeze_dim_17: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_145, 0);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_133: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_17, [1, 16, 1024, 64]);  squeeze_dim_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_55: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_134: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_55, [1, 1024, 1024]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_135: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_134, [1024, 1024]);  view_134 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_39: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg110_1, view_135, permute_74);  arg110_1 = view_135 = permute_74 = None
    view_136: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_39, [1, 1024, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_48: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_45, view_136);  add_45 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_22: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_48, getitem_29);  getitem_29 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_54: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_55: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_54, arg111_1);  mul_54 = arg111_1 = None
    add_50: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_55, arg112_1);  mul_55 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_137: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_50, [1024, 1024]);  add_50 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    addmm_40: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg114_1, view_137, permute_75);  arg114_1 = view_137 = permute_75 = None
    view_138: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_40, [1, 1024, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_56: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
    mul_57: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
    erf_6: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_51: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_58: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_56, add_51);  mul_56 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_139: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_58, [1024, 4096]);  mul_58 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_41: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg116_1, view_139, permute_76);  arg116_1 = view_139 = permute_76 = None
    view_140: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_41, [1, 1024, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_52: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_48, view_140);  add_48 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_23: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_52, getitem_31);  getitem_31 = None
    add_53: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_59: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_60: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_59, arg117_1);  mul_59 = arg117_1 = None
    add_54: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_60, arg118_1);  mul_60 = arg118_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_141: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_54, [1024, 1024])
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_42: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg120_1, view_141, permute_77);  arg120_1 = view_141 = permute_77 = None
    view_142: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_42, [1, 1024, 1024]);  addmm_42 = None
    mul_61: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_61, [1, 1024, 16, 64]);  mul_61 = None
    permute_82: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_61: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_150: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_61, [16, -1, 64]);  clone_61 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_48: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_150, 0);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_143: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_54, [1024, 1024])
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_43: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg122_1, view_143, permute_78);  arg122_1 = view_143 = permute_78 = None
    view_144: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_43, [1, 1024, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_144, [1, -1, 16, 64]);  view_144 = None
    permute_79: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_59: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_151: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_59, [16, -1, 64]);  clone_59 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_49: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_151, 0);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_146: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_54, [1024, 1024]);  add_54 = None
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    addmm_44: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg124_1, view_146, permute_80);  arg124_1 = view_146 = permute_80 = None
    view_147: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_44, [1, 1024, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_147, [1, -1, 16, 64]);  view_147 = None
    permute_81: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_60: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_152: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_60, [16, -1, 64]);  clone_60 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_50: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_152, 0);  view_152 = None
    _scaled_dot_product_flash_attention_default_16 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_48, unsqueeze_default_49, unsqueeze_default_50, scale = 1.0);  unsqueeze_default_48 = unsqueeze_default_49 = unsqueeze_default_50 = None
    getitem_144: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_16[0];  _scaled_dot_product_flash_attention_default_16 = None
    squeeze_dim_16: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_144, 0);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_153: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_16, [1, 16, 1024, 64]);  squeeze_dim_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_63: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_154: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_63, [1, 1024, 1024]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_155: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_154, [1024, 1024]);  view_154 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_45: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg126_1, view_155, permute_85);  arg126_1 = view_155 = permute_85 = None
    view_156: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_45, [1, 1024, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_55: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_52, view_156);  add_52 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_25: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_55, getitem_33);  getitem_33 = None
    add_56: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_62: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
    mul_63: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_62, arg127_1);  mul_62 = arg127_1 = None
    add_57: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_63, arg128_1);  mul_63 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_157: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_57, [1024, 1024]);  add_57 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_46: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg130_1, view_157, permute_86);  arg130_1 = view_157 = permute_86 = None
    view_158: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_46, [1, 1024, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_64: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_65: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_7: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_58: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_66: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_64, add_58);  mul_64 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_159: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_66, [1024, 4096]);  mul_66 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_47: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg132_1, view_159, permute_87);  arg132_1 = view_159 = permute_87 = None
    view_160: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_47, [1, 1024, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_59: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_55, view_160);  add_55 = view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_26: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_59, getitem_35);  getitem_35 = None
    add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_67: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_68: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_67, arg133_1);  mul_67 = arg133_1 = None
    add_61: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_68, arg134_1);  mul_68 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_161: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_61, [1024, 1024])
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_48: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg136_1, view_161, permute_88);  arg136_1 = view_161 = permute_88 = None
    view_162: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_48, [1, 1024, 1024]);  addmm_48 = None
    mul_69: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_162, 0.125);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_69, [1, 1024, 16, 64]);  mul_69 = None
    permute_93: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
    clone_69: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_170: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_69, [16, -1, 64]);  clone_69 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_45: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_170, 0);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_163: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_61, [1024, 1024])
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_49: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_163, permute_89);  arg138_1 = view_163 = permute_89 = None
    view_164: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_49, [1, 1024, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_164, [1, -1, 16, 64]);  view_164 = None
    permute_90: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    clone_67: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_171: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_67, [16, -1, 64]);  clone_67 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_46: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_171, 0);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_166: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_61, [1024, 1024]);  add_61 = None
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_50: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg140_1, view_166, permute_91);  arg140_1 = view_166 = permute_91 = None
    view_167: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_50, [1, 1024, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_168: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_167, [1, -1, 16, 64]);  view_167 = None
    permute_92: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_68: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_172: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_68, [16, -1, 64]);  clone_68 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_47: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_172, 0);  view_172 = None
    _scaled_dot_product_flash_attention_default_15 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_45, unsqueeze_default_46, unsqueeze_default_47, scale = 1.0);  unsqueeze_default_45 = unsqueeze_default_46 = unsqueeze_default_47 = None
    getitem_143: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_15[0];  _scaled_dot_product_flash_attention_default_15 = None
    squeeze_dim_15: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_143, 0);  getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_173: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_15, [1, 16, 1024, 64]);  squeeze_dim_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_71: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_174: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_71, [1, 1024, 1024]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_175: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_174, [1024, 1024]);  view_174 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_51: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg142_1, view_175, permute_96);  arg142_1 = view_175 = permute_96 = None
    view_176: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_51, [1, 1024, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_62: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_59, view_176);  add_59 = view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_28: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_62, getitem_37);  getitem_37 = None
    add_63: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    mul_70: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
    mul_71: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_70, arg143_1);  mul_70 = arg143_1 = None
    add_64: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_71, arg144_1);  mul_71 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_177: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_64, [1024, 1024]);  add_64 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    addmm_52: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg146_1, view_177, permute_97);  arg146_1 = view_177 = permute_97 = None
    view_178: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_52, [1, 1024, 4096]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_72: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    mul_73: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
    erf_8: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
    add_65: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_74: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_72, add_65);  mul_72 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_179: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_74, [1024, 4096]);  mul_74 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_53: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg148_1, view_179, permute_98);  arg148_1 = view_179 = permute_98 = None
    view_180: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_53, [1, 1024, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_66: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_62, view_180);  add_62 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_29: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_66, getitem_39);  getitem_39 = None
    add_67: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_75: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_76: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_75, arg149_1);  mul_75 = arg149_1 = None
    add_68: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_76, arg150_1);  mul_76 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_181: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_68, [1024, 1024])
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_54: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg152_1, view_181, permute_99);  arg152_1 = view_181 = permute_99 = None
    view_182: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_54, [1, 1024, 1024]);  addmm_54 = None
    mul_77: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_182, 0.125);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_189: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_77, [1, 1024, 16, 64]);  mul_77 = None
    permute_104: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
    clone_77: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_190: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_77, [16, -1, 64]);  clone_77 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_42: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_190, 0);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_183: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_68, [1024, 1024])
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_55: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg154_1, view_183, permute_100);  arg154_1 = view_183 = permute_100 = None
    view_184: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_55, [1, 1024, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_185: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_184, [1, -1, 16, 64]);  view_184 = None
    permute_101: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    clone_75: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_191: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_75, [16, -1, 64]);  clone_75 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_43: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_191, 0);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_186: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_68, [1024, 1024]);  add_68 = None
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_56: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg156_1, view_186, permute_102);  arg156_1 = view_186 = permute_102 = None
    view_187: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_56, [1, 1024, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_188: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_187, [1, -1, 16, 64]);  view_187 = None
    permute_103: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    clone_76: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_192: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_76, [16, -1, 64]);  clone_76 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_44: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_192, 0);  view_192 = None
    _scaled_dot_product_flash_attention_default_14 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_42, unsqueeze_default_43, unsqueeze_default_44, scale = 1.0);  unsqueeze_default_42 = unsqueeze_default_43 = unsqueeze_default_44 = None
    getitem_142: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_14[0];  _scaled_dot_product_flash_attention_default_14 = None
    squeeze_dim_14: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_142, 0);  getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_193: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_14, [1, 16, 1024, 64]);  squeeze_dim_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_79: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_194: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_79, [1, 1024, 1024]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_195: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_194, [1024, 1024]);  view_194 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_57: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg158_1, view_195, permute_107);  arg158_1 = view_195 = permute_107 = None
    view_196: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_57, [1, 1024, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_69: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_66, view_196);  add_66 = view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_31: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_69, getitem_41);  getitem_41 = None
    add_70: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    mul_78: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
    mul_79: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_78, arg159_1);  mul_78 = arg159_1 = None
    add_71: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_79, arg160_1);  mul_79 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_197: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_71, [1024, 1024]);  add_71 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    addmm_58: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg162_1, view_197, permute_108);  arg162_1 = view_197 = permute_108 = None
    view_198: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_58, [1, 1024, 4096]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_80: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_198, 0.5)
    mul_81: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476);  view_198 = None
    erf_9: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_72: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_82: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_80, add_72);  mul_80 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_199: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_82, [1024, 4096]);  mul_82 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_59: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg164_1, view_199, permute_109);  arg164_1 = view_199 = permute_109 = None
    view_200: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_59, [1, 1024, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_73: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_69, view_200);  add_69 = view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_32: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_43);  getitem_43 = None
    add_74: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_83: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_84: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_83, arg165_1);  mul_83 = arg165_1 = None
    add_75: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_84, arg166_1);  mul_84 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_201: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_75, [1024, 1024])
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_60: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_201, permute_110);  arg168_1 = view_201 = permute_110 = None
    view_202: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_60, [1, 1024, 1024]);  addmm_60 = None
    mul_85: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_202, 0.125);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_209: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_85, [1, 1024, 16, 64]);  mul_85 = None
    permute_115: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_85: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_210: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_85, [16, -1, 64]);  clone_85 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_39: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_210, 0);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_203: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_75, [1024, 1024])
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_61: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_203, permute_111);  arg170_1 = view_203 = permute_111 = None
    view_204: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_61, [1, 1024, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_205: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_204, [1, -1, 16, 64]);  view_204 = None
    permute_112: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    clone_83: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_211: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_83, [16, -1, 64]);  clone_83 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_40: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_211, 0);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_206: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_75, [1024, 1024]);  add_75 = None
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_62: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg172_1, view_206, permute_113);  arg172_1 = view_206 = permute_113 = None
    view_207: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_62, [1, 1024, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_208: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_207, [1, -1, 16, 64]);  view_207 = None
    permute_114: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_84: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_212: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_84, [16, -1, 64]);  clone_84 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_41: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_212, 0);  view_212 = None
    _scaled_dot_product_flash_attention_default_13 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_39, unsqueeze_default_40, unsqueeze_default_41, scale = 1.0);  unsqueeze_default_39 = unsqueeze_default_40 = unsqueeze_default_41 = None
    getitem_141: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_13[0];  _scaled_dot_product_flash_attention_default_13 = None
    squeeze_dim_13: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_141, 0);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_213: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_13, [1, 16, 1024, 64]);  squeeze_dim_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_87: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_214: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_87, [1, 1024, 1024]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_215: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_214, [1024, 1024]);  view_214 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_63: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg174_1, view_215, permute_118);  arg174_1 = view_215 = permute_118 = None
    view_216: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_63, [1, 1024, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_76: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_73, view_216);  add_73 = view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_34: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_76, getitem_45);  getitem_45 = None
    add_77: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    mul_86: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_87: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_86, arg175_1);  mul_86 = arg175_1 = None
    add_78: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_87, arg176_1);  mul_87 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_217: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_78, [1024, 1024]);  add_78 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    addmm_64: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg178_1, view_217, permute_119);  arg178_1 = view_217 = permute_119 = None
    view_218: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_64, [1, 1024, 4096]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_218, 0.5)
    mul_89: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_218, 0.7071067811865476);  view_218 = None
    erf_10: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_79: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_90: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_88, add_79);  mul_88 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_219: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_90, [1024, 4096]);  mul_90 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_65: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg180_1, view_219, permute_120);  arg180_1 = view_219 = permute_120 = None
    view_220: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_65, [1, 1024, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_80: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_76, view_220);  add_76 = view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_35: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_80, getitem_47);  getitem_47 = None
    add_81: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_91: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_92: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_91, arg181_1);  mul_91 = arg181_1 = None
    add_82: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_92, arg182_1);  mul_92 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_221: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_82, [1024, 1024])
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_66: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg184_1, view_221, permute_121);  arg184_1 = view_221 = permute_121 = None
    view_222: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_66, [1, 1024, 1024]);  addmm_66 = None
    mul_93: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_222, 0.125);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_229: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_93, [1, 1024, 16, 64]);  mul_93 = None
    permute_126: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    clone_93: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_230: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_93, [16, -1, 64]);  clone_93 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_36: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_230, 0);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_223: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_82, [1024, 1024])
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_67: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg186_1, view_223, permute_122);  arg186_1 = view_223 = permute_122 = None
    view_224: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_67, [1, 1024, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_225: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_224, [1, -1, 16, 64]);  view_224 = None
    permute_123: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    clone_91: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_231: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_91, [16, -1, 64]);  clone_91 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_37: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_231, 0);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_226: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_82, [1024, 1024]);  add_82 = None
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_68: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg188_1, view_226, permute_124);  arg188_1 = view_226 = permute_124 = None
    view_227: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_68, [1, 1024, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_228: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_227, [1, -1, 16, 64]);  view_227 = None
    permute_125: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    clone_92: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_232: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_92, [16, -1, 64]);  clone_92 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_38: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_232, 0);  view_232 = None
    _scaled_dot_product_flash_attention_default_12 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_36, unsqueeze_default_37, unsqueeze_default_38, scale = 1.0);  unsqueeze_default_36 = unsqueeze_default_37 = unsqueeze_default_38 = None
    getitem_140: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_12[0];  _scaled_dot_product_flash_attention_default_12 = None
    squeeze_dim_12: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_140, 0);  getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_233: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_12, [1, 16, 1024, 64]);  squeeze_dim_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_95: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_234: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_95, [1, 1024, 1024]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_235: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_234, [1024, 1024]);  view_234 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_69: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg190_1, view_235, permute_129);  arg190_1 = view_235 = permute_129 = None
    view_236: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_69, [1, 1024, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_83: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_80, view_236);  add_80 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 1024, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_37: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_83, getitem_49);  getitem_49 = None
    add_84: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    mul_94: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
    mul_95: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_94, arg191_1);  mul_94 = arg191_1 = None
    add_85: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_95, arg192_1);  mul_95 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_237: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_85, [1024, 1024]);  add_85 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    addmm_70: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg194_1, view_237, permute_130);  arg194_1 = view_237 = permute_130 = None
    view_238: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_70, [1, 1024, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_96: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_238, 0.5)
    mul_97: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_238, 0.7071067811865476);  view_238 = None
    erf_11: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_86: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_98: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_96, add_86);  mul_96 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_239: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_98, [1024, 4096]);  mul_98 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_71: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg196_1, view_239, permute_131);  arg196_1 = view_239 = permute_131 = None
    view_240: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_71, [1, 1024, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_87: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_83, view_240);  add_83 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:857, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 1024, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 1024, 1]" = var_mean_25[1];  var_mean_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:71, code: prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(arg518_1, -100)
    full_default: "i64[]" = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "i64[1, 1024]" = torch.ops.aten.where.self(eq, full_default, arg518_1);  eq = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:75, code: prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    slice_8: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(where, 1, 1, 9223372036854775807)
    slice_4: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(where, 1, 0, -1)
    clone_1: "i64[1, 1023]" = torch.ops.aten.clone.default(slice_4);  slice_4 = None
    copy: "i64[1, 1023]" = torch.ops.aten.copy.default(slice_8, clone_1);  slice_8 = clone_1 = None
    slice_scatter: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(where, copy, 1, 1, 9223372036854775807);  copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:76, code: prev_output_tokens[:, 0] = decoder_start_tokens
    select_1: "i64[1]" = torch.ops.aten.select.int(slice_scatter, 1, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:73, code: index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    ne: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(where, 1)
    sum_1: "i64[1]" = torch.ops.aten.sum.dim_IntList(ne, [1]);  ne = None
    sub: "i64[1]" = torch.ops.aten.sub.Tensor(sum_1, 1);  sum_1 = None
    unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(sub, -1);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:74, code: decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    gather: "i64[1, 1]" = torch.ops.aten.gather.default(where, 1, unsqueeze);  where = unsqueeze = None
    
    # No stacktrace found for following nodes
    squeeze: "i64[]" = torch.ops.aten.squeeze.default(gather);  gather = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:76, code: prev_output_tokens[:, 0] = decoder_start_tokens
    copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, squeeze);  select_1 = squeeze = None
    select_scatter: "i64[1, 1024]" = torch.ops.aten.select_scatter.default(slice_scatter, copy_1, 1, 0);  slice_scatter = copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1037, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    view_242: "i64[1, 1024]" = torch.ops.aten.reshape.default(select_scatter, [-1, 1024]);  select_scatter = None
    embedding_2: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg199_1, view_242, 1);  arg199_1 = view_242 = None
    mul_101: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:130, code: positions = torch.arange(
    iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:132, code: ).expand(bsz, -1)
    expand_2: "i64[1, 1024]" = torch.ops.aten.expand.default(iota_2, [1, -1]);  iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:134, code: return super().forward(positions + self.offset)
    add_91: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand_2, 2);  expand_2 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, add_91);  arg1_1 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1051, code: hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
    add_92: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_101, embedding_3);  mul_101 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1052, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 1024, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 1024, 1]" = var_mean_26[1];  var_mean_26 = None
    sub_39: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_92, getitem_53);  add_92 = getitem_53 = None
    add_93: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_102: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
    mul_103: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_102, arg200_1);  mul_102 = arg200_1 = None
    add_94: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_103, arg201_1);  mul_103 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 1024, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 1024, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_40: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_94, getitem_55);  getitem_55 = None
    add_95: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    mul_104: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_27);  sub_40 = rsqrt_27 = None
    mul_105: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_104, arg202_1);  mul_104 = arg202_1 = None
    add_96: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_105, arg203_1);  mul_105 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_244: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_96, [1024, 1024])
    permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_72: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg205_1, view_244, permute_132);  arg205_1 = view_244 = permute_132 = None
    view_245: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_72, [1, 1024, 1024]);  addmm_72 = None
    mul_106: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_245, 0.125);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_252: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_106, [1, 1024, 16, 64]);  mul_106 = None
    permute_137: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    clone_102: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_253: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_102, [16, -1, 64]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_246: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_96, [1024, 1024])
    permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_73: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg207_1, view_246, permute_133);  arg207_1 = view_246 = permute_133 = None
    view_247: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_73, [1, 1024, 1024]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_248: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_247, [1, -1, 16, 64]);  view_247 = None
    permute_134: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    clone_100: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_254: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_100, [16, -1, 64]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_138: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_24: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_253, permute_138);  view_253 = permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_256: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 1024, 1024]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:90, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:91, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_90: "i64[1024]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_243: "i64[1024, 1]" = torch.ops.aten.reshape.default(add_90, [1024, 1]);  add_90 = None
    lt: "b8[1024, 1024]" = torch.ops.aten.lt.Tensor(iota_1, view_243);  iota_1 = view_243 = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:89, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default_1: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:91, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where_1: "f32[1024, 1024]" = torch.ops.aten.where.self(lt, full_default_2, full_default_1);  lt = full_default_2 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_3: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where_1, 0);  where_1 = None
    unsqueeze_4: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
    expand_3: "f32[1, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_4, [1, 1, 1024, 1024]);  unsqueeze_4 = None
    add_97: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_256, expand_3);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_257: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_97, [16, 1024, 1024]);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_257, [-1], True)
    sub_41: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_257, amax_12);  view_257 = amax_12 = None
    exp_12: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_14: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_14);  exp_12 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_249: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_96, [1024, 1024]);  add_96 = None
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_74: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg209_1, view_249, permute_135);  arg209_1 = view_249 = permute_135 = None
    view_250: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_74, [1, 1024, 1024]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_251: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_250, [1, -1, 16, 64]);  view_250 = None
    permute_136: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    clone_101: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_255: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_101, [16, -1, 64]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_12, view_255);  div_12 = view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_258: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 1024, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_139: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_104: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_259: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_104, [1, 1024, 1024]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_260: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_259, [1024, 1024]);  view_259 = None
    permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_75: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg211_1, view_260, permute_140);  arg211_1 = view_260 = permute_140 = None
    view_261: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_75, [1, 1024, 1024]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_98: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_94, view_261);  add_94 = view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 1024, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 1024, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_42: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_98, getitem_57);  getitem_57 = None
    add_99: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    mul_107: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
    mul_108: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_107, arg212_1);  mul_107 = arg212_1 = None
    add_100: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_108, arg213_1);  mul_108 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_262: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_100, [1024, 1024]);  add_100 = None
    permute_141: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_76: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg215_1, view_262, permute_141);  arg215_1 = view_262 = permute_141 = None
    view_263: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_76, [1, 1024, 1024]);  addmm_76 = None
    mul_109: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.125);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_270: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_109, [1, 1024, 16, 64]);  mul_109 = None
    permute_146: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
    clone_108: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_271: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_108, [16, -1, 64]);  clone_108 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_33: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_271, 0);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:857, code: hidden_states = self.layer_norm(hidden_states)
    sub_38: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_87, getitem_51);  add_87 = getitem_51 = None
    add_88: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    mul_99: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_100: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_99, arg197_1);  mul_99 = arg197_1 = None
    add_89: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_100, arg198_1);  mul_100 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_264: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_142: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    addmm_77: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg217_1, view_264, permute_142);  arg217_1 = view_264 = permute_142 = None
    view_265: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_77, [1, 1024, 1024]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_266: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_265, [1, -1, 16, 64]);  view_265 = None
    permute_143: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
    clone_106: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_272: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_106, [16, -1, 64]);  clone_106 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_34: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_272, 0);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_267: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    addmm_78: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg219_1, view_267, permute_144);  arg219_1 = view_267 = permute_144 = None
    view_268: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_78, [1, 1024, 1024]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_269: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_268, [1, -1, 16, 64]);  view_268 = None
    permute_145: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_107: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_273: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_107, [16, -1, 64]);  clone_107 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_35: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_273, 0);  view_273 = None
    _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
    getitem_139: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_11[0];  _scaled_dot_product_flash_attention_default_11 = None
    squeeze_dim_11: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_139, 0);  getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_274: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_11, [1, 16, 1024, 64]);  squeeze_dim_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_148: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_110: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_275: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_110, [1, 1024, 1024]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_276: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_275, [1024, 1024]);  view_275 = None
    permute_149: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    addmm_79: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg221_1, view_276, permute_149);  arg221_1 = view_276 = permute_149 = None
    view_277: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_79, [1, 1024, 1024]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_101: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_98, view_277);  add_98 = view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 1024, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 1024, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_44: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_101, getitem_59);  getitem_59 = None
    add_102: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    mul_110: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_111: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_110, arg222_1);  mul_110 = arg222_1 = None
    add_103: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_111, arg223_1);  mul_111 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_278: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_103, [1024, 1024]);  add_103 = None
    permute_150: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    addmm_80: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg225_1, view_278, permute_150);  arg225_1 = view_278 = permute_150 = None
    view_279: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_80, [1, 1024, 4096]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_112: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
    mul_113: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476);  view_279 = None
    erf_12: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_104: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_114: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_112, add_104);  mul_112 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_280: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_114, [1024, 4096]);  mul_114 = None
    permute_151: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    addmm_81: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg227_1, view_280, permute_151);  arg227_1 = view_280 = permute_151 = None
    view_281: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_81, [1, 1024, 1024]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_105: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_101, view_281);  add_101 = view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 1024, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 1024, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_45: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_61);  getitem_61 = None
    add_106: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    mul_115: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
    mul_116: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_115, arg228_1);  mul_115 = arg228_1 = None
    add_107: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_116, arg229_1);  mul_116 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_282: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_107, [1024, 1024])
    permute_152: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_82: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg231_1, view_282, permute_152);  arg231_1 = view_282 = permute_152 = None
    view_283: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_82, [1, 1024, 1024]);  addmm_82 = None
    mul_117: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_283, 0.125);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_290: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_117, [1, 1024, 16, 64]);  mul_117 = None
    permute_157: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    clone_116: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_291: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_116, [16, -1, 64]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_284: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_107, [1024, 1024])
    permute_153: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
    addmm_83: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg233_1, view_284, permute_153);  arg233_1 = view_284 = permute_153 = None
    view_285: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_83, [1, 1024, 1024]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_286: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_285, [1, -1, 16, 64]);  view_285 = None
    permute_154: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    clone_114: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_292: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_114, [16, -1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_158: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
    bmm_28: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_291, permute_158);  view_291 = permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_294: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 1024, 1024]);  bmm_28 = None
    add_108: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_294, expand_3);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_295: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_108, [16, 1024, 1024]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_295, [-1], True)
    sub_46: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_295, amax_14);  view_295 = amax_14 = None
    exp_14: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_16);  exp_14 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_287: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_107, [1024, 1024]);  add_107 = None
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_84: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg235_1, view_287, permute_155);  arg235_1 = view_287 = permute_155 = None
    view_288: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_84, [1, 1024, 1024]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_289: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_288, [1, -1, 16, 64]);  view_288 = None
    permute_156: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    clone_115: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_293: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_115, [16, -1, 64]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_14, view_293);  div_14 = view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_296: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 1024, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_159: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_118: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    view_297: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_118, [1, 1024, 1024]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_298: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_297, [1024, 1024]);  view_297 = None
    permute_160: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    addmm_85: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg237_1, view_298, permute_160);  arg237_1 = view_298 = permute_160 = None
    view_299: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_85, [1, 1024, 1024]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_109: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_105, view_299);  add_105 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 1024, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 1024, 1]" = var_mean_31[1];  var_mean_31 = None
    sub_47: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_63);  getitem_63 = None
    add_110: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    mul_118: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_119: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_118, arg238_1);  mul_118 = arg238_1 = None
    add_111: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_119, arg239_1);  mul_119 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_300: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_111, [1024, 1024]);  add_111 = None
    permute_161: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_86: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg241_1, view_300, permute_161);  arg241_1 = view_300 = permute_161 = None
    view_301: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_86, [1, 1024, 1024]);  addmm_86 = None
    mul_120: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_301, 0.125);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_308: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_120, [1, 1024, 16, 64]);  mul_120 = None
    permute_166: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    clone_122: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_309: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_122, [16, -1, 64]);  clone_122 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_30: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_309, 0);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_302: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
    addmm_87: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg243_1, view_302, permute_162);  arg243_1 = view_302 = permute_162 = None
    view_303: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_87, [1, 1024, 1024]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_304: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_303, [1, -1, 16, 64]);  view_303 = None
    permute_163: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
    clone_120: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_310: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_120, [16, -1, 64]);  clone_120 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_31: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_310, 0);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_305: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_164: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    addmm_88: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg245_1, view_305, permute_164);  arg245_1 = view_305 = permute_164 = None
    view_306: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_88, [1, 1024, 1024]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_307: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_306, [1, -1, 16, 64]);  view_306 = None
    permute_165: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    clone_121: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_311: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_121, [16, -1, 64]);  clone_121 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_32: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_311, 0);  view_311 = None
    _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
    getitem_138: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_10[0];  _scaled_dot_product_flash_attention_default_10 = None
    squeeze_dim_10: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_138, 0);  getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_312: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_10, [1, 16, 1024, 64]);  squeeze_dim_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_168: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_124: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_313: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_124, [1, 1024, 1024]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_314: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_313, [1024, 1024]);  view_313 = None
    permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    addmm_89: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg247_1, view_314, permute_169);  arg247_1 = view_314 = permute_169 = None
    view_315: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_89, [1, 1024, 1024]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_112: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_109, view_315);  add_109 = view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 1024, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 1024, 1]" = var_mean_32[1];  var_mean_32 = None
    sub_49: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_112, getitem_65);  getitem_65 = None
    add_113: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    mul_121: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
    mul_122: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_121, arg248_1);  mul_121 = arg248_1 = None
    add_114: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_122, arg249_1);  mul_122 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_316: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_114, [1024, 1024]);  add_114 = None
    permute_170: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_90: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg251_1, view_316, permute_170);  arg251_1 = view_316 = permute_170 = None
    view_317: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_90, [1, 1024, 4096]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_123: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_317, 0.5)
    mul_124: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_317, 0.7071067811865476);  view_317 = None
    erf_13: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_124);  mul_124 = None
    add_115: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_125: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_123, add_115);  mul_123 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_318: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_125, [1024, 4096]);  mul_125 = None
    permute_171: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
    addmm_91: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg253_1, view_318, permute_171);  arg253_1 = view_318 = permute_171 = None
    view_319: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_91, [1, 1024, 1024]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_116: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_112, view_319);  add_112 = view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 1024, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 1024, 1]" = var_mean_33[1];  var_mean_33 = None
    sub_50: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_116, getitem_67);  getitem_67 = None
    add_117: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    mul_126: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
    mul_127: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_126, arg254_1);  mul_126 = arg254_1 = None
    add_118: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_127, arg255_1);  mul_127 = arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_320: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_118, [1024, 1024])
    permute_172: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    addmm_92: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg257_1, view_320, permute_172);  arg257_1 = view_320 = permute_172 = None
    view_321: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_92, [1, 1024, 1024]);  addmm_92 = None
    mul_128: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_321, 0.125);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_328: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_128, [1, 1024, 16, 64]);  mul_128 = None
    permute_177: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
    clone_130: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_329: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_130, [16, -1, 64]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_322: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_118, [1024, 1024])
    permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
    addmm_93: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg259_1, view_322, permute_173);  arg259_1 = view_322 = permute_173 = None
    view_323: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_93, [1, 1024, 1024]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_324: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_323, [1, -1, 16, 64]);  view_323 = None
    permute_174: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    clone_128: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_330: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_128, [16, -1, 64]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_178: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None
    bmm_32: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_329, permute_178);  view_329 = permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_332: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 1024, 1024]);  bmm_32 = None
    add_119: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_332, expand_3);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_333: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_119, [16, 1024, 1024]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_333, [-1], True)
    sub_51: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_333, amax_16);  view_333 = amax_16 = None
    exp_16: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_18: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_18);  exp_16 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_325: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_118, [1024, 1024]);  add_118 = None
    permute_175: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm_94: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg261_1, view_325, permute_175);  arg261_1 = view_325 = permute_175 = None
    view_326: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_94, [1, 1024, 1024]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_327: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_326, [1, -1, 16, 64]);  view_326 = None
    permute_176: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    clone_129: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_331: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_129, [16, -1, 64]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_16, view_331);  div_16 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_334: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 1024, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_179: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_132: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_335: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_132, [1, 1024, 1024]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_336: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_335, [1024, 1024]);  view_335 = None
    permute_180: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
    addmm_95: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg263_1, view_336, permute_180);  arg263_1 = view_336 = permute_180 = None
    view_337: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_95, [1, 1024, 1024]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_120: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_116, view_337);  add_116 = view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1024, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 1024, 1]" = var_mean_34[1];  var_mean_34 = None
    sub_52: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_120, getitem_69);  getitem_69 = None
    add_121: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    mul_129: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
    mul_130: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_129, arg264_1);  mul_129 = arg264_1 = None
    add_122: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_130, arg265_1);  mul_130 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_338: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_122, [1024, 1024]);  add_122 = None
    permute_181: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    addmm_96: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg267_1, view_338, permute_181);  arg267_1 = view_338 = permute_181 = None
    view_339: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_96, [1, 1024, 1024]);  addmm_96 = None
    mul_131: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_339, 0.125);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_346: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_131, [1, 1024, 16, 64]);  mul_131 = None
    permute_186: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    clone_136: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_347: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_136, [16, -1, 64]);  clone_136 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_27: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_347, 0);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_340: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_182: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
    addmm_97: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg269_1, view_340, permute_182);  arg269_1 = view_340 = permute_182 = None
    view_341: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_97, [1, 1024, 1024]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_342: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_341, [1, -1, 16, 64]);  view_341 = None
    permute_183: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    clone_134: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_348: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_134, [16, -1, 64]);  clone_134 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_28: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_348, 0);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_343: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_98: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg271_1, view_343, permute_184);  arg271_1 = view_343 = permute_184 = None
    view_344: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_98, [1, 1024, 1024]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_345: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_344, [1, -1, 16, 64]);  view_344 = None
    permute_185: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_135: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_349: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_135, [16, -1, 64]);  clone_135 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_29: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_349, 0);  view_349 = None
    _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
    getitem_137: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_9[0];  _scaled_dot_product_flash_attention_default_9 = None
    squeeze_dim_9: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_137, 0);  getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_350: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_9, [1, 16, 1024, 64]);  squeeze_dim_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_188: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_138: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_351: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_138, [1, 1024, 1024]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_352: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_351, [1024, 1024]);  view_351 = None
    permute_189: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
    addmm_99: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg273_1, view_352, permute_189);  arg273_1 = view_352 = permute_189 = None
    view_353: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_99, [1, 1024, 1024]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_123: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_120, view_353);  add_120 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_123, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 1024, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 1024, 1]" = var_mean_35[1];  var_mean_35 = None
    sub_54: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_123, getitem_71);  getitem_71 = None
    add_124: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    mul_132: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = rsqrt_35 = None
    mul_133: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_132, arg274_1);  mul_132 = arg274_1 = None
    add_125: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_133, arg275_1);  mul_133 = arg275_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_354: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_125, [1024, 1024]);  add_125 = None
    permute_190: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    addmm_100: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg277_1, view_354, permute_190);  arg277_1 = view_354 = permute_190 = None
    view_355: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_100, [1, 1024, 4096]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_134: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_355, 0.5)
    mul_135: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_355, 0.7071067811865476);  view_355 = None
    erf_14: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_126: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_136: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_134, add_126);  mul_134 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_356: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_136, [1024, 4096]);  mul_136 = None
    permute_191: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    addmm_101: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg279_1, view_356, permute_191);  arg279_1 = view_356 = permute_191 = None
    view_357: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_101, [1, 1024, 1024]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_127: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_123, view_357);  add_123 = view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_127, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 1024, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 1024, 1]" = var_mean_36[1];  var_mean_36 = None
    sub_55: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_127, getitem_73);  getitem_73 = None
    add_128: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    mul_137: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = rsqrt_36 = None
    mul_138: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_137, arg280_1);  mul_137 = arg280_1 = None
    add_129: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_138, arg281_1);  mul_138 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_358: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_129, [1024, 1024])
    permute_192: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
    addmm_102: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg283_1, view_358, permute_192);  arg283_1 = view_358 = permute_192 = None
    view_359: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_102, [1, 1024, 1024]);  addmm_102 = None
    mul_139: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_359, 0.125);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_366: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_139, [1, 1024, 16, 64]);  mul_139 = None
    permute_197: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    clone_144: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_367: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_144, [16, -1, 64]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_360: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_129, [1024, 1024])
    permute_193: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
    addmm_103: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg285_1, view_360, permute_193);  arg285_1 = view_360 = permute_193 = None
    view_361: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_103, [1, 1024, 1024]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_362: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_361, [1, -1, 16, 64]);  view_361 = None
    permute_194: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
    clone_142: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_368: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_142, [16, -1, 64]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_198: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
    bmm_36: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_367, permute_198);  view_367 = permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_370: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 1024, 1024]);  bmm_36 = None
    add_130: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_370, expand_3);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_371: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_130, [16, 1024, 1024]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_371, [-1], True)
    sub_56: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_371, amax_18);  view_371 = amax_18 = None
    exp_18: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_20: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_18, sum_20);  exp_18 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_363: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_129, [1024, 1024]);  add_129 = None
    permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
    addmm_104: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg287_1, view_363, permute_195);  arg287_1 = view_363 = permute_195 = None
    view_364: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_104, [1, 1024, 1024]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_365: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_364, [1, -1, 16, 64]);  view_364 = None
    permute_196: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    clone_143: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_369: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_143, [16, -1, 64]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_37: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_18, view_369);  div_18 = view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_372: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 1024, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_199: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_146: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_373: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_146, [1, 1024, 1024]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_374: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_373, [1024, 1024]);  view_373 = None
    permute_200: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
    addmm_105: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg289_1, view_374, permute_200);  arg289_1 = view_374 = permute_200 = None
    view_375: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_105, [1, 1024, 1024]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_131: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_127, view_375);  add_127 = view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 1024, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 1024, 1]" = var_mean_37[1];  var_mean_37 = None
    sub_57: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_131, getitem_75);  getitem_75 = None
    add_132: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    mul_140: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = rsqrt_37 = None
    mul_141: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_140, arg290_1);  mul_140 = arg290_1 = None
    add_133: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_141, arg291_1);  mul_141 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_376: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_133, [1024, 1024]);  add_133 = None
    permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
    addmm_106: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg293_1, view_376, permute_201);  arg293_1 = view_376 = permute_201 = None
    view_377: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_106, [1, 1024, 1024]);  addmm_106 = None
    mul_142: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_377, 0.125);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_384: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_142, [1, 1024, 16, 64]);  mul_142 = None
    permute_206: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    clone_150: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_385: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_150, [16, -1, 64]);  clone_150 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_24: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_385, 0);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_378: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_202: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
    addmm_107: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg295_1, view_378, permute_202);  arg295_1 = view_378 = permute_202 = None
    view_379: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_107, [1, 1024, 1024]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_380: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_379, [1, -1, 16, 64]);  view_379 = None
    permute_203: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
    clone_148: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_386: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_148, [16, -1, 64]);  clone_148 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_25: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_386, 0);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_381: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_204: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
    addmm_108: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg297_1, view_381, permute_204);  arg297_1 = view_381 = permute_204 = None
    view_382: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_108, [1, 1024, 1024]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_383: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_382, [1, -1, 16, 64]);  view_382 = None
    permute_205: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    clone_149: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_387: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_149, [16, -1, 64]);  clone_149 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_26: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_387, 0);  view_387 = None
    _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
    getitem_136: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_8[0];  _scaled_dot_product_flash_attention_default_8 = None
    squeeze_dim_8: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_136, 0);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_388: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_8, [1, 16, 1024, 64]);  squeeze_dim_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_208: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_152: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    view_389: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_152, [1, 1024, 1024]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_390: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_389, [1024, 1024]);  view_389 = None
    permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
    addmm_109: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg299_1, view_390, permute_209);  arg299_1 = view_390 = permute_209 = None
    view_391: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_109, [1, 1024, 1024]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_134: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_131, view_391);  add_131 = view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1024, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 1024, 1]" = var_mean_38[1];  var_mean_38 = None
    sub_59: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_134, getitem_77);  getitem_77 = None
    add_135: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    mul_143: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_38);  sub_59 = rsqrt_38 = None
    mul_144: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_143, arg300_1);  mul_143 = arg300_1 = None
    add_136: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_144, arg301_1);  mul_144 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_392: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_136, [1024, 1024]);  add_136 = None
    permute_210: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
    addmm_110: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg303_1, view_392, permute_210);  arg303_1 = view_392 = permute_210 = None
    view_393: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_110, [1, 1024, 4096]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_145: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
    mul_146: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
    erf_15: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_137: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_147: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_145, add_137);  mul_145 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_394: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_147, [1024, 4096]);  mul_147 = None
    permute_211: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
    addmm_111: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg305_1, view_394, permute_211);  arg305_1 = view_394 = permute_211 = None
    view_395: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_111, [1, 1024, 1024]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_138: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_134, view_395);  add_134 = view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_138, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 1024, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 1024, 1]" = var_mean_39[1];  var_mean_39 = None
    sub_60: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_138, getitem_79);  getitem_79 = None
    add_139: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    mul_148: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = rsqrt_39 = None
    mul_149: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_148, arg306_1);  mul_148 = arg306_1 = None
    add_140: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_149, arg307_1);  mul_149 = arg307_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_396: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_140, [1024, 1024])
    permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
    addmm_112: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg309_1, view_396, permute_212);  arg309_1 = view_396 = permute_212 = None
    view_397: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_112, [1, 1024, 1024]);  addmm_112 = None
    mul_150: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_397, 0.125);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_404: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_150, [1, 1024, 16, 64]);  mul_150 = None
    permute_217: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    clone_158: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_405: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_158, [16, -1, 64]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_398: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_140, [1024, 1024])
    permute_213: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
    addmm_113: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg311_1, view_398, permute_213);  arg311_1 = view_398 = permute_213 = None
    view_399: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_113, [1, 1024, 1024]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_400: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_399, [1, -1, 16, 64]);  view_399 = None
    permute_214: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    clone_156: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_406: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_156, [16, -1, 64]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_218: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
    bmm_40: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_405, permute_218);  view_405 = permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_408: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 1024, 1024]);  bmm_40 = None
    add_141: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_408, expand_3);  view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_409: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_141, [16, 1024, 1024]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_409, [-1], True)
    sub_61: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_409, amax_20);  view_409 = amax_20 = None
    exp_20: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_22: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_20, sum_22);  exp_20 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_401: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_140, [1024, 1024]);  add_140 = None
    permute_215: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
    addmm_114: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg313_1, view_401, permute_215);  arg313_1 = view_401 = permute_215 = None
    view_402: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_114, [1, 1024, 1024]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_403: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_402, [1, -1, 16, 64]);  view_402 = None
    permute_216: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    clone_157: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_407: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_157, [16, -1, 64]);  clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_41: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_20, view_407);  div_20 = view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_410: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 1024, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_219: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_160: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_411: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_160, [1, 1024, 1024]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_412: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_411, [1024, 1024]);  view_411 = None
    permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
    addmm_115: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg315_1, view_412, permute_220);  arg315_1 = view_412 = permute_220 = None
    view_413: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_115, [1, 1024, 1024]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_142: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_138, view_413);  add_138 = view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 1024, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 1024, 1]" = var_mean_40[1];  var_mean_40 = None
    sub_62: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_142, getitem_81);  getitem_81 = None
    add_143: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    mul_151: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_40);  sub_62 = rsqrt_40 = None
    mul_152: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_151, arg316_1);  mul_151 = arg316_1 = None
    add_144: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_152, arg317_1);  mul_152 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_414: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_144, [1024, 1024]);  add_144 = None
    permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
    addmm_116: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg319_1, view_414, permute_221);  arg319_1 = view_414 = permute_221 = None
    view_415: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_116, [1, 1024, 1024]);  addmm_116 = None
    mul_153: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_415, 0.125);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_422: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_153, [1, 1024, 16, 64]);  mul_153 = None
    permute_226: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    clone_164: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_423: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_164, [16, -1, 64]);  clone_164 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_21: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_423, 0);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_416: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_222: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
    addmm_117: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg321_1, view_416, permute_222);  arg321_1 = view_416 = permute_222 = None
    view_417: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_117, [1, 1024, 1024]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_418: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_417, [1, -1, 16, 64]);  view_417 = None
    permute_223: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
    clone_162: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_424: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_162, [16, -1, 64]);  clone_162 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_22: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_424, 0);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_419: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_224: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
    addmm_118: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg323_1, view_419, permute_224);  arg323_1 = view_419 = permute_224 = None
    view_420: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_118, [1, 1024, 1024]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_421: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_420, [1, -1, 16, 64]);  view_420 = None
    permute_225: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    clone_163: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_425: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_163, [16, -1, 64]);  clone_163 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_23: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_425, 0);  view_425 = None
    _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
    getitem_135: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_7[0];  _scaled_dot_product_flash_attention_default_7 = None
    squeeze_dim_7: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_135, 0);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_426: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_7, [1, 16, 1024, 64]);  squeeze_dim_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_228: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_166: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_427: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_166, [1, 1024, 1024]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_428: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_427, [1024, 1024]);  view_427 = None
    permute_229: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
    addmm_119: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg325_1, view_428, permute_229);  arg325_1 = view_428 = permute_229 = None
    view_429: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_119, [1, 1024, 1024]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_145: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_142, view_429);  add_142 = view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 1024, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 1024, 1]" = var_mean_41[1];  var_mean_41 = None
    sub_64: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_145, getitem_83);  getitem_83 = None
    add_146: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    mul_154: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_41);  sub_64 = rsqrt_41 = None
    mul_155: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_154, arg326_1);  mul_154 = arg326_1 = None
    add_147: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_155, arg327_1);  mul_155 = arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_430: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_147, [1024, 1024]);  add_147 = None
    permute_230: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
    addmm_120: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg329_1, view_430, permute_230);  arg329_1 = view_430 = permute_230 = None
    view_431: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_120, [1, 1024, 4096]);  addmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_156: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_431, 0.5)
    mul_157: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_431, 0.7071067811865476);  view_431 = None
    erf_16: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_157);  mul_157 = None
    add_148: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_158: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_156, add_148);  mul_156 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_432: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_158, [1024, 4096]);  mul_158 = None
    permute_231: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
    addmm_121: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg331_1, view_432, permute_231);  arg331_1 = view_432 = permute_231 = None
    view_433: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_121, [1, 1024, 1024]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_149: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_145, view_433);  add_145 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1024, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1024, 1]" = var_mean_42[1];  var_mean_42 = None
    sub_65: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_85);  getitem_85 = None
    add_150: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    mul_159: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_42);  sub_65 = rsqrt_42 = None
    mul_160: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_159, arg332_1);  mul_159 = arg332_1 = None
    add_151: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_160, arg333_1);  mul_160 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_434: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_151, [1024, 1024])
    permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
    addmm_122: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg335_1, view_434, permute_232);  arg335_1 = view_434 = permute_232 = None
    view_435: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_122, [1, 1024, 1024]);  addmm_122 = None
    mul_161: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_435, 0.125);  view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_442: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_161, [1, 1024, 16, 64]);  mul_161 = None
    permute_237: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    clone_172: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_443: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_172, [16, -1, 64]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_436: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_151, [1024, 1024])
    permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
    addmm_123: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg337_1, view_436, permute_233);  arg337_1 = view_436 = permute_233 = None
    view_437: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_123, [1, 1024, 1024]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_438: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_437, [1, -1, 16, 64]);  view_437 = None
    permute_234: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None
    clone_170: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_444: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_170, [16, -1, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_238: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
    bmm_44: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_443, permute_238);  view_443 = permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_446: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 1024, 1024]);  bmm_44 = None
    add_152: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_446, expand_3);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_447: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_152, [16, 1024, 1024]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_447, [-1], True)
    sub_66: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_447, amax_22);  view_447 = amax_22 = None
    exp_22: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_24: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_22, sum_24);  exp_22 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_439: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_151, [1024, 1024]);  add_151 = None
    permute_235: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
    addmm_124: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg339_1, view_439, permute_235);  arg339_1 = view_439 = permute_235 = None
    view_440: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_124, [1, 1024, 1024]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_441: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_440, [1, -1, 16, 64]);  view_440 = None
    permute_236: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
    clone_171: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_445: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_171, [16, -1, 64]);  clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_45: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_22, view_445);  div_22 = view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_448: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 1024, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_239: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_174: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
    view_449: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_174, [1, 1024, 1024]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_450: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_449, [1024, 1024]);  view_449 = None
    permute_240: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
    addmm_125: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg341_1, view_450, permute_240);  arg341_1 = view_450 = permute_240 = None
    view_451: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_125, [1, 1024, 1024]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_153: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_149, view_451);  add_149 = view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1024, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1024, 1]" = var_mean_43[1];  var_mean_43 = None
    sub_67: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_87);  getitem_87 = None
    add_154: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    mul_162: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_43);  sub_67 = rsqrt_43 = None
    mul_163: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_162, arg342_1);  mul_162 = arg342_1 = None
    add_155: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_163, arg343_1);  mul_163 = arg343_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_452: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_155, [1024, 1024]);  add_155 = None
    permute_241: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
    addmm_126: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg345_1, view_452, permute_241);  arg345_1 = view_452 = permute_241 = None
    view_453: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_126, [1, 1024, 1024]);  addmm_126 = None
    mul_164: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_453, 0.125);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_460: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_164, [1, 1024, 16, 64]);  mul_164 = None
    permute_246: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
    clone_178: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_461: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_178, [16, -1, 64]);  clone_178 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_18: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_461, 0);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_454: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
    addmm_127: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg347_1, view_454, permute_242);  arg347_1 = view_454 = permute_242 = None
    view_455: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_127, [1, 1024, 1024]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_456: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_455, [1, -1, 16, 64]);  view_455 = None
    permute_243: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    clone_176: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_462: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_176, [16, -1, 64]);  clone_176 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_19: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_462, 0);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_457: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_244: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
    addmm_128: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg349_1, view_457, permute_244);  arg349_1 = view_457 = permute_244 = None
    view_458: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_128, [1, 1024, 1024]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_459: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_458, [1, -1, 16, 64]);  view_458 = None
    permute_245: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    clone_177: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_463: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_177, [16, -1, 64]);  clone_177 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_20: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_463, 0);  view_463 = None
    _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
    getitem_134: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_6[0];  _scaled_dot_product_flash_attention_default_6 = None
    squeeze_dim_6: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_134, 0);  getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_464: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_6, [1, 16, 1024, 64]);  squeeze_dim_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_248: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_180: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    view_465: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_180, [1, 1024, 1024]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_466: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_465, [1024, 1024]);  view_465 = None
    permute_249: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
    addmm_129: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg351_1, view_466, permute_249);  arg351_1 = view_466 = permute_249 = None
    view_467: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_129, [1, 1024, 1024]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_156: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_153, view_467);  add_153 = view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_156, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 1024, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 1024, 1]" = var_mean_44[1];  var_mean_44 = None
    sub_69: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_156, getitem_89);  getitem_89 = None
    add_157: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    mul_165: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_44);  sub_69 = rsqrt_44 = None
    mul_166: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_165, arg352_1);  mul_165 = arg352_1 = None
    add_158: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_166, arg353_1);  mul_166 = arg353_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_468: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_158, [1024, 1024]);  add_158 = None
    permute_250: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
    addmm_130: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg355_1, view_468, permute_250);  arg355_1 = view_468 = permute_250 = None
    view_469: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_130, [1, 1024, 4096]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_167: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_469, 0.5)
    mul_168: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_469, 0.7071067811865476);  view_469 = None
    erf_17: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_159: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_169: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_167, add_159);  mul_167 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_470: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_169, [1024, 4096]);  mul_169 = None
    permute_251: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
    addmm_131: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg357_1, view_470, permute_251);  arg357_1 = view_470 = permute_251 = None
    view_471: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_131, [1, 1024, 1024]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_160: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_156, view_471);  add_156 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 1024, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 1024, 1]" = var_mean_45[1];  var_mean_45 = None
    sub_70: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_160, getitem_91);  getitem_91 = None
    add_161: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    mul_170: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_45);  sub_70 = rsqrt_45 = None
    mul_171: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_170, arg358_1);  mul_170 = arg358_1 = None
    add_162: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_171, arg359_1);  mul_171 = arg359_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_472: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_162, [1024, 1024])
    permute_252: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
    addmm_132: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg361_1, view_472, permute_252);  arg361_1 = view_472 = permute_252 = None
    view_473: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_132, [1, 1024, 1024]);  addmm_132 = None
    mul_172: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_473, 0.125);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_480: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_172, [1, 1024, 16, 64]);  mul_172 = None
    permute_257: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
    clone_186: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_481: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_186, [16, -1, 64]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_474: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_162, [1024, 1024])
    permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
    addmm_133: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg363_1, view_474, permute_253);  arg363_1 = view_474 = permute_253 = None
    view_475: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_133, [1, 1024, 1024]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_476: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_475, [1, -1, 16, 64]);  view_475 = None
    permute_254: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    clone_184: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_482: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_184, [16, -1, 64]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_258: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_482, [0, 2, 1]);  view_482 = None
    bmm_48: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_481, permute_258);  view_481 = permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_484: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_48, [1, 16, 1024, 1024]);  bmm_48 = None
    add_163: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_484, expand_3);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_485: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_163, [16, 1024, 1024]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_24: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_485, [-1], True)
    sub_71: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_485, amax_24);  view_485 = amax_24 = None
    exp_24: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
    sum_26: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_24: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_24, sum_26);  exp_24 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_477: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_162, [1024, 1024]);  add_162 = None
    permute_255: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
    addmm_134: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg365_1, view_477, permute_255);  arg365_1 = view_477 = permute_255 = None
    view_478: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_134, [1, 1024, 1024]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_479: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_478, [1, -1, 16, 64]);  view_478 = None
    permute_256: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
    clone_185: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_483: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_185, [16, -1, 64]);  clone_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_49: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_24, view_483);  div_24 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_486: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_49, [1, 16, 1024, 64]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_259: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_188: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_487: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_188, [1, 1024, 1024]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_488: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_487, [1024, 1024]);  view_487 = None
    permute_260: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg366_1, [1, 0]);  arg366_1 = None
    addmm_135: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg367_1, view_488, permute_260);  arg367_1 = view_488 = permute_260 = None
    view_489: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_135, [1, 1024, 1024]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_164: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_160, view_489);  add_160 = view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 1024, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 1024, 1]" = var_mean_46[1];  var_mean_46 = None
    sub_72: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_164, getitem_93);  getitem_93 = None
    add_165: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    mul_173: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_46);  sub_72 = rsqrt_46 = None
    mul_174: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_173, arg368_1);  mul_173 = arg368_1 = None
    add_166: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_174, arg369_1);  mul_174 = arg369_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_490: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_166, [1024, 1024]);  add_166 = None
    permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
    addmm_136: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg371_1, view_490, permute_261);  arg371_1 = view_490 = permute_261 = None
    view_491: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_136, [1, 1024, 1024]);  addmm_136 = None
    mul_175: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_491, 0.125);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_498: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_175, [1, 1024, 16, 64]);  mul_175 = None
    permute_266: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    clone_192: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_499: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_192, [16, -1, 64]);  clone_192 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_15: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_499, 0);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_492: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_262: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg372_1, [1, 0]);  arg372_1 = None
    addmm_137: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg373_1, view_492, permute_262);  arg373_1 = view_492 = permute_262 = None
    view_493: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_137, [1, 1024, 1024]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_494: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_493, [1, -1, 16, 64]);  view_493 = None
    permute_263: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    clone_190: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_500: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_190, [16, -1, 64]);  clone_190 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_16: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_500, 0);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_495: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
    addmm_138: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg375_1, view_495, permute_264);  arg375_1 = view_495 = permute_264 = None
    view_496: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_138, [1, 1024, 1024]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_497: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_496, [1, -1, 16, 64]);  view_496 = None
    permute_265: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_191: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_501: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_191, [16, -1, 64]);  clone_191 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_17: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_501, 0);  view_501 = None
    _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
    getitem_133: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_5[0];  _scaled_dot_product_flash_attention_default_5 = None
    squeeze_dim_5: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_133, 0);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_502: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_5, [1, 16, 1024, 64]);  squeeze_dim_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_268: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_194: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    view_503: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_194, [1, 1024, 1024]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_504: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_503, [1024, 1024]);  view_503 = None
    permute_269: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
    addmm_139: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg377_1, view_504, permute_269);  arg377_1 = view_504 = permute_269 = None
    view_505: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_139, [1, 1024, 1024]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_167: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_164, view_505);  add_164 = view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 1024, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 1024, 1]" = var_mean_47[1];  var_mean_47 = None
    sub_74: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_167, getitem_95);  getitem_95 = None
    add_168: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    mul_176: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_47);  sub_74 = rsqrt_47 = None
    mul_177: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_176, arg378_1);  mul_176 = arg378_1 = None
    add_169: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_177, arg379_1);  mul_177 = arg379_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_506: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_169, [1024, 1024]);  add_169 = None
    permute_270: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
    addmm_140: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg381_1, view_506, permute_270);  arg381_1 = view_506 = permute_270 = None
    view_507: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_140, [1, 1024, 4096]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_178: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_507, 0.5)
    mul_179: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_507, 0.7071067811865476);  view_507 = None
    erf_18: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
    add_170: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_180: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_178, add_170);  mul_178 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_508: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_180, [1024, 4096]);  mul_180 = None
    permute_271: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
    addmm_141: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg383_1, view_508, permute_271);  arg383_1 = view_508 = permute_271 = None
    view_509: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_141, [1, 1024, 1024]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_171: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_167, view_509);  add_167 = view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 1024, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 1024, 1]" = var_mean_48[1];  var_mean_48 = None
    sub_75: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_171, getitem_97);  getitem_97 = None
    add_172: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    mul_181: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_48);  sub_75 = rsqrt_48 = None
    mul_182: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_181, arg384_1);  mul_181 = arg384_1 = None
    add_173: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_182, arg385_1);  mul_182 = arg385_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_510: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_173, [1024, 1024])
    permute_272: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
    addmm_142: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg387_1, view_510, permute_272);  arg387_1 = view_510 = permute_272 = None
    view_511: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_142, [1, 1024, 1024]);  addmm_142 = None
    mul_183: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_511, 0.125);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_518: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_183, [1, 1024, 16, 64]);  mul_183 = None
    permute_277: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
    clone_200: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_519: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_200, [16, -1, 64]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_512: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_173, [1024, 1024])
    permute_273: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
    addmm_143: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg389_1, view_512, permute_273);  arg389_1 = view_512 = permute_273 = None
    view_513: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_143, [1, 1024, 1024]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_514: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_513, [1, -1, 16, 64]);  view_513 = None
    permute_274: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    clone_198: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_520: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_198, [16, -1, 64]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_278: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
    bmm_52: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_519, permute_278);  view_519 = permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_522: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_52, [1, 16, 1024, 1024]);  bmm_52 = None
    add_174: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_522, expand_3);  view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_523: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_174, [16, 1024, 1024]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_26: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_523, [-1], True)
    sub_76: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_523, amax_26);  view_523 = amax_26 = None
    exp_26: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_28: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_26: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_26, sum_28);  exp_26 = sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_515: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_173, [1024, 1024]);  add_173 = None
    permute_275: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
    addmm_144: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg391_1, view_515, permute_275);  arg391_1 = view_515 = permute_275 = None
    view_516: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_144, [1, 1024, 1024]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_517: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_516, [1, -1, 16, 64]);  view_516 = None
    permute_276: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    clone_199: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_521: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_199, [16, -1, 64]);  clone_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_53: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_26, view_521);  div_26 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_524: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_53, [1, 16, 1024, 64]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_279: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_202: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_525: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_202, [1, 1024, 1024]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_526: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_525, [1024, 1024]);  view_525 = None
    permute_280: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
    addmm_145: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg393_1, view_526, permute_280);  arg393_1 = view_526 = permute_280 = None
    view_527: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_145, [1, 1024, 1024]);  addmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_175: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_171, view_527);  add_171 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 1024, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 1024, 1]" = var_mean_49[1];  var_mean_49 = None
    sub_77: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_175, getitem_99);  getitem_99 = None
    add_176: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    mul_184: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_49);  sub_77 = rsqrt_49 = None
    mul_185: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_184, arg394_1);  mul_184 = arg394_1 = None
    add_177: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_185, arg395_1);  mul_185 = arg395_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_528: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_177, [1024, 1024]);  add_177 = None
    permute_281: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
    addmm_146: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg397_1, view_528, permute_281);  arg397_1 = view_528 = permute_281 = None
    view_529: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_146, [1, 1024, 1024]);  addmm_146 = None
    mul_186: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.125);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_536: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_186, [1, 1024, 16, 64]);  mul_186 = None
    permute_286: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_536, [0, 2, 1, 3]);  view_536 = None
    clone_206: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_537: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_206, [16, -1, 64]);  clone_206 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_12: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_537, 0);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_530: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_282: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg398_1, [1, 0]);  arg398_1 = None
    addmm_147: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg399_1, view_530, permute_282);  arg399_1 = view_530 = permute_282 = None
    view_531: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_147, [1, 1024, 1024]);  addmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_532: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_531, [1, -1, 16, 64]);  view_531 = None
    permute_283: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    clone_204: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_538: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_204, [16, -1, 64]);  clone_204 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_13: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_538, 0);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_533: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_284: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg400_1, [1, 0]);  arg400_1 = None
    addmm_148: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg401_1, view_533, permute_284);  arg401_1 = view_533 = permute_284 = None
    view_534: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_148, [1, 1024, 1024]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_535: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_534, [1, -1, 16, 64]);  view_534 = None
    permute_285: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
    clone_205: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_539: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_205, [16, -1, 64]);  clone_205 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_14: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_539, 0);  view_539 = None
    _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
    getitem_132: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_4[0];  _scaled_dot_product_flash_attention_default_4 = None
    squeeze_dim_4: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_132, 0);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_540: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_4, [1, 16, 1024, 64]);  squeeze_dim_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_288: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_208: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_541: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_208, [1, 1024, 1024]);  clone_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_542: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_541, [1024, 1024]);  view_541 = None
    permute_289: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg402_1, [1, 0]);  arg402_1 = None
    addmm_149: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg403_1, view_542, permute_289);  arg403_1 = view_542 = permute_289 = None
    view_543: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_149, [1, 1024, 1024]);  addmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_178: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_175, view_543);  add_175 = view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_50 = torch.ops.aten.var_mean.correction(add_178, [2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 1024, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 1024, 1]" = var_mean_50[1];  var_mean_50 = None
    sub_79: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_178, getitem_101);  getitem_101 = None
    add_179: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    mul_187: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_50);  sub_79 = rsqrt_50 = None
    mul_188: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_187, arg404_1);  mul_187 = arg404_1 = None
    add_180: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_188, arg405_1);  mul_188 = arg405_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_544: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_180, [1024, 1024]);  add_180 = None
    permute_290: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg406_1, [1, 0]);  arg406_1 = None
    addmm_150: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg407_1, view_544, permute_290);  arg407_1 = view_544 = permute_290 = None
    view_545: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_150, [1, 1024, 4096]);  addmm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_189: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_545, 0.5)
    mul_190: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_545, 0.7071067811865476);  view_545 = None
    erf_19: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
    add_181: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_191: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_189, add_181);  mul_189 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_546: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_191, [1024, 4096]);  mul_191 = None
    permute_291: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg408_1, [1, 0]);  arg408_1 = None
    addmm_151: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg409_1, view_546, permute_291);  arg409_1 = view_546 = permute_291 = None
    view_547: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_151, [1, 1024, 1024]);  addmm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_182: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_178, view_547);  add_178 = view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_51 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1024, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1024, 1]" = var_mean_51[1];  var_mean_51 = None
    sub_80: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_182, getitem_103);  getitem_103 = None
    add_183: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    mul_192: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_51);  sub_80 = rsqrt_51 = None
    mul_193: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_192, arg410_1);  mul_192 = arg410_1 = None
    add_184: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_193, arg411_1);  mul_193 = arg411_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_548: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_184, [1024, 1024])
    permute_292: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg412_1, [1, 0]);  arg412_1 = None
    addmm_152: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg413_1, view_548, permute_292);  arg413_1 = view_548 = permute_292 = None
    view_549: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_152, [1, 1024, 1024]);  addmm_152 = None
    mul_194: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_549, 0.125);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_556: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_194, [1, 1024, 16, 64]);  mul_194 = None
    permute_297: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
    clone_214: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_557: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_214, [16, -1, 64]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_550: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_184, [1024, 1024])
    permute_293: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg414_1, [1, 0]);  arg414_1 = None
    addmm_153: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg415_1, view_550, permute_293);  arg415_1 = view_550 = permute_293 = None
    view_551: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_153, [1, 1024, 1024]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_552: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_551, [1, -1, 16, 64]);  view_551 = None
    permute_294: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None
    clone_212: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_558: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_212, [16, -1, 64]);  clone_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_298: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_558, [0, 2, 1]);  view_558 = None
    bmm_56: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_557, permute_298);  view_557 = permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_560: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_56, [1, 16, 1024, 1024]);  bmm_56 = None
    add_185: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_560, expand_3);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_561: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_185, [16, 1024, 1024]);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_28: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_561, [-1], True)
    sub_81: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_561, amax_28);  view_561 = amax_28 = None
    exp_28: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_81);  sub_81 = None
    sum_30: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
    div_28: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_28, sum_30);  exp_28 = sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_553: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_184, [1024, 1024]);  add_184 = None
    permute_295: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
    addmm_154: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg417_1, view_553, permute_295);  arg417_1 = view_553 = permute_295 = None
    view_554: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_154, [1, 1024, 1024]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_555: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_554, [1, -1, 16, 64]);  view_554 = None
    permute_296: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
    clone_213: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_559: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_213, [16, -1, 64]);  clone_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_57: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_28, view_559);  div_28 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_562: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_57, [1, 16, 1024, 64]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_299: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_216: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_563: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_216, [1, 1024, 1024]);  clone_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_564: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_563, [1024, 1024]);  view_563 = None
    permute_300: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg418_1, [1, 0]);  arg418_1 = None
    addmm_155: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg419_1, view_564, permute_300);  arg419_1 = view_564 = permute_300 = None
    view_565: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_155, [1, 1024, 1024]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_186: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_182, view_565);  add_182 = view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_52 = torch.ops.aten.var_mean.correction(add_186, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 1024, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 1024, 1]" = var_mean_52[1];  var_mean_52 = None
    sub_82: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_186, getitem_105);  getitem_105 = None
    add_187: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    mul_195: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_52);  sub_82 = rsqrt_52 = None
    mul_196: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_195, arg420_1);  mul_195 = arg420_1 = None
    add_188: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_196, arg421_1);  mul_196 = arg421_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_566: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_188, [1024, 1024]);  add_188 = None
    permute_301: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg422_1, [1, 0]);  arg422_1 = None
    addmm_156: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg423_1, view_566, permute_301);  arg423_1 = view_566 = permute_301 = None
    view_567: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_156, [1, 1024, 1024]);  addmm_156 = None
    mul_197: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_567, 0.125);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_574: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_197, [1, 1024, 16, 64]);  mul_197 = None
    permute_306: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    clone_220: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_575: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_220, [16, -1, 64]);  clone_220 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_9: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_575, 0);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_568: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_302: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg424_1, [1, 0]);  arg424_1 = None
    addmm_157: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg425_1, view_568, permute_302);  arg425_1 = view_568 = permute_302 = None
    view_569: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_157, [1, 1024, 1024]);  addmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_570: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_569, [1, -1, 16, 64]);  view_569 = None
    permute_303: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    clone_218: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_576: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_218, [16, -1, 64]);  clone_218 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_10: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_576, 0);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_571: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_304: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg426_1, [1, 0]);  arg426_1 = None
    addmm_158: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg427_1, view_571, permute_304);  arg427_1 = view_571 = permute_304 = None
    view_572: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_158, [1, 1024, 1024]);  addmm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_573: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_572, [1, -1, 16, 64]);  view_572 = None
    permute_305: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    clone_219: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_577: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_219, [16, -1, 64]);  clone_219 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_11: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_577, 0);  view_577 = None
    _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
    getitem_131: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_3[0];  _scaled_dot_product_flash_attention_default_3 = None
    squeeze_dim_3: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_131, 0);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_578: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_3, [1, 16, 1024, 64]);  squeeze_dim_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_308: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_222: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    view_579: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_222, [1, 1024, 1024]);  clone_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_580: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_579, [1024, 1024]);  view_579 = None
    permute_309: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg428_1, [1, 0]);  arg428_1 = None
    addmm_159: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg429_1, view_580, permute_309);  arg429_1 = view_580 = permute_309 = None
    view_581: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_159, [1, 1024, 1024]);  addmm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_189: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_186, view_581);  add_186 = view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_53 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 1024, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 1024, 1]" = var_mean_53[1];  var_mean_53 = None
    sub_84: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_189, getitem_107);  getitem_107 = None
    add_190: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    mul_198: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_53);  sub_84 = rsqrt_53 = None
    mul_199: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_198, arg430_1);  mul_198 = arg430_1 = None
    add_191: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_199, arg431_1);  mul_199 = arg431_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_582: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_191, [1024, 1024]);  add_191 = None
    permute_310: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg432_1, [1, 0]);  arg432_1 = None
    addmm_160: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg433_1, view_582, permute_310);  arg433_1 = view_582 = permute_310 = None
    view_583: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_160, [1, 1024, 4096]);  addmm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_200: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_583, 0.5)
    mul_201: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_583, 0.7071067811865476);  view_583 = None
    erf_20: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_192: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_202: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_200, add_192);  mul_200 = add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_584: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_202, [1024, 4096]);  mul_202 = None
    permute_311: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
    addmm_161: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg435_1, view_584, permute_311);  arg435_1 = view_584 = permute_311 = None
    view_585: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_161, [1, 1024, 1024]);  addmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_193: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_189, view_585);  add_189 = view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_54 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 1024, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 1024, 1]" = var_mean_54[1];  var_mean_54 = None
    sub_85: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_109);  getitem_109 = None
    add_194: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    mul_203: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_54);  sub_85 = rsqrt_54 = None
    mul_204: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_203, arg436_1);  mul_203 = arg436_1 = None
    add_195: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_204, arg437_1);  mul_204 = arg437_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_586: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_195, [1024, 1024])
    permute_312: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg438_1, [1, 0]);  arg438_1 = None
    addmm_162: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg439_1, view_586, permute_312);  arg439_1 = view_586 = permute_312 = None
    view_587: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_162, [1, 1024, 1024]);  addmm_162 = None
    mul_205: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_587, 0.125);  view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_594: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_205, [1, 1024, 16, 64]);  mul_205 = None
    permute_317: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    clone_228: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_595: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_228, [16, -1, 64]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_588: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_195, [1024, 1024])
    permute_313: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg440_1, [1, 0]);  arg440_1 = None
    addmm_163: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg441_1, view_588, permute_313);  arg441_1 = view_588 = permute_313 = None
    view_589: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_163, [1, 1024, 1024]);  addmm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_590: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_589, [1, -1, 16, 64]);  view_589 = None
    permute_314: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_590, [0, 2, 1, 3]);  view_590 = None
    clone_226: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_596: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_226, [16, -1, 64]);  clone_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_318: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_596, [0, 2, 1]);  view_596 = None
    bmm_60: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_595, permute_318);  view_595 = permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_598: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_60, [1, 16, 1024, 1024]);  bmm_60 = None
    add_196: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_598, expand_3);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_599: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_196, [16, 1024, 1024]);  add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_30: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_599, [-1], True)
    sub_86: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_599, amax_30);  view_599 = amax_30 = None
    exp_30: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_86);  sub_86 = None
    sum_32: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
    div_30: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_30, sum_32);  exp_30 = sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_591: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_195, [1024, 1024]);  add_195 = None
    permute_315: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg442_1, [1, 0]);  arg442_1 = None
    addmm_164: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg443_1, view_591, permute_315);  arg443_1 = view_591 = permute_315 = None
    view_592: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_164, [1, 1024, 1024]);  addmm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_593: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_592, [1, -1, 16, 64]);  view_592 = None
    permute_316: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    clone_227: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_597: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_227, [16, -1, 64]);  clone_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_61: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_30, view_597);  div_30 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_600: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_61, [1, 16, 1024, 64]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_319: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_230: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_601: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_230, [1, 1024, 1024]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_602: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_601, [1024, 1024]);  view_601 = None
    permute_320: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg444_1, [1, 0]);  arg444_1 = None
    addmm_165: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg445_1, view_602, permute_320);  arg445_1 = view_602 = permute_320 = None
    view_603: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_165, [1, 1024, 1024]);  addmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_197: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_193, view_603);  add_193 = view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_55 = torch.ops.aten.var_mean.correction(add_197, [2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1024, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 1024, 1]" = var_mean_55[1];  var_mean_55 = None
    sub_87: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_197, getitem_111);  getitem_111 = None
    add_198: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    mul_206: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_55);  sub_87 = rsqrt_55 = None
    mul_207: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_206, arg446_1);  mul_206 = arg446_1 = None
    add_199: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_207, arg447_1);  mul_207 = arg447_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_604: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_199, [1024, 1024]);  add_199 = None
    permute_321: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg448_1, [1, 0]);  arg448_1 = None
    addmm_166: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg449_1, view_604, permute_321);  arg449_1 = view_604 = permute_321 = None
    view_605: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_166, [1, 1024, 1024]);  addmm_166 = None
    mul_208: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_605, 0.125);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_612: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_208, [1, 1024, 16, 64]);  mul_208 = None
    permute_326: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    clone_234: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_613: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_234, [16, -1, 64]);  clone_234 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_6: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_613, 0);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_606: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_322: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
    addmm_167: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg451_1, view_606, permute_322);  arg451_1 = view_606 = permute_322 = None
    view_607: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_167, [1, 1024, 1024]);  addmm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_608: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_607, [1, -1, 16, 64]);  view_607 = None
    permute_323: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
    clone_232: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_614: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_232, [16, -1, 64]);  clone_232 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_7: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_614, 0);  view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_609: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_324: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
    addmm_168: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg453_1, view_609, permute_324);  arg453_1 = view_609 = permute_324 = None
    view_610: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_168, [1, 1024, 1024]);  addmm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_611: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_610, [1, -1, 16, 64]);  view_610 = None
    permute_325: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None
    clone_233: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_615: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_233, [16, -1, 64]);  clone_233 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_8: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_615, 0);  view_615 = None
    _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
    getitem_130: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_2[0];  _scaled_dot_product_flash_attention_default_2 = None
    squeeze_dim_2: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_130, 0);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_616: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_2, [1, 16, 1024, 64]);  squeeze_dim_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_328: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_236: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_617: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_236, [1, 1024, 1024]);  clone_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_618: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_617, [1024, 1024]);  view_617 = None
    permute_329: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg454_1, [1, 0]);  arg454_1 = None
    addmm_169: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg455_1, view_618, permute_329);  arg455_1 = view_618 = permute_329 = None
    view_619: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_169, [1, 1024, 1024]);  addmm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_200: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_197, view_619);  add_197 = view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_56 = torch.ops.aten.var_mean.correction(add_200, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 1024, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 1024, 1]" = var_mean_56[1];  var_mean_56 = None
    sub_89: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_200, getitem_113);  getitem_113 = None
    add_201: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    mul_209: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_56);  sub_89 = rsqrt_56 = None
    mul_210: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_209, arg456_1);  mul_209 = arg456_1 = None
    add_202: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_210, arg457_1);  mul_210 = arg457_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_620: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_202, [1024, 1024]);  add_202 = None
    permute_330: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg458_1, [1, 0]);  arg458_1 = None
    addmm_170: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg459_1, view_620, permute_330);  arg459_1 = view_620 = permute_330 = None
    view_621: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_170, [1, 1024, 4096]);  addmm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_211: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_621, 0.5)
    mul_212: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_621, 0.7071067811865476);  view_621 = None
    erf_21: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_212);  mul_212 = None
    add_203: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_213: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_211, add_203);  mul_211 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_622: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_213, [1024, 4096]);  mul_213 = None
    permute_331: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg460_1, [1, 0]);  arg460_1 = None
    addmm_171: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg461_1, view_622, permute_331);  arg461_1 = view_622 = permute_331 = None
    view_623: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_171, [1, 1024, 1024]);  addmm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_204: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_200, view_623);  add_200 = view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_57 = torch.ops.aten.var_mean.correction(add_204, [2], correction = 0, keepdim = True)
    getitem_114: "f32[1, 1024, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 1024, 1]" = var_mean_57[1];  var_mean_57 = None
    sub_90: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_204, getitem_115);  getitem_115 = None
    add_205: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_57: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    mul_214: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_57);  sub_90 = rsqrt_57 = None
    mul_215: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_214, arg462_1);  mul_214 = arg462_1 = None
    add_206: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_215, arg463_1);  mul_215 = arg463_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_624: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_206, [1024, 1024])
    permute_332: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg464_1, [1, 0]);  arg464_1 = None
    addmm_172: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg465_1, view_624, permute_332);  arg465_1 = view_624 = permute_332 = None
    view_625: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_172, [1, 1024, 1024]);  addmm_172 = None
    mul_216: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_625, 0.125);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_632: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_216, [1, 1024, 16, 64]);  mul_216 = None
    permute_337: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
    clone_242: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_633: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_242, [16, -1, 64]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_626: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_206, [1024, 1024])
    permute_333: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg466_1, [1, 0]);  arg466_1 = None
    addmm_173: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg467_1, view_626, permute_333);  arg467_1 = view_626 = permute_333 = None
    view_627: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_173, [1, 1024, 1024]);  addmm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_628: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_627, [1, -1, 16, 64]);  view_627 = None
    permute_334: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_628, [0, 2, 1, 3]);  view_628 = None
    clone_240: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_634: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_240, [16, -1, 64]);  clone_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_338: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_634, [0, 2, 1]);  view_634 = None
    bmm_64: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_633, permute_338);  view_633 = permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_636: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_64, [1, 16, 1024, 1024]);  bmm_64 = None
    add_207: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_636, expand_3);  view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_637: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_207, [16, 1024, 1024]);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_32: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_637, [-1], True)
    sub_91: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_637, amax_32);  view_637 = amax_32 = None
    exp_32: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_91);  sub_91 = None
    sum_34: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
    div_32: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_32, sum_34);  exp_32 = sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_629: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_206, [1024, 1024]);  add_206 = None
    permute_335: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
    addmm_174: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg469_1, view_629, permute_335);  arg469_1 = view_629 = permute_335 = None
    view_630: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_174, [1, 1024, 1024]);  addmm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_631: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_630, [1, -1, 16, 64]);  view_630 = None
    permute_336: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
    clone_241: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_635: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_241, [16, -1, 64]);  clone_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_65: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_32, view_635);  div_32 = view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_638: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_65, [1, 16, 1024, 64]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_339: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_244: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_339, memory_format = torch.contiguous_format);  permute_339 = None
    view_639: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_244, [1, 1024, 1024]);  clone_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_640: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_639, [1024, 1024]);  view_639 = None
    permute_340: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
    addmm_175: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg471_1, view_640, permute_340);  arg471_1 = view_640 = permute_340 = None
    view_641: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_175, [1, 1024, 1024]);  addmm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_208: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_204, view_641);  add_204 = view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_58 = torch.ops.aten.var_mean.correction(add_208, [2], correction = 0, keepdim = True)
    getitem_116: "f32[1, 1024, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 1024, 1]" = var_mean_58[1];  var_mean_58 = None
    sub_92: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_208, getitem_117);  getitem_117 = None
    add_209: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_58: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    mul_217: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_58);  sub_92 = rsqrt_58 = None
    mul_218: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_217, arg472_1);  mul_217 = arg472_1 = None
    add_210: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_218, arg473_1);  mul_218 = arg473_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_642: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_210, [1024, 1024]);  add_210 = None
    permute_341: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg474_1, [1, 0]);  arg474_1 = None
    addmm_176: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg475_1, view_642, permute_341);  arg475_1 = view_642 = permute_341 = None
    view_643: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_176, [1, 1024, 1024]);  addmm_176 = None
    mul_219: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_643, 0.125);  view_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_650: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_219, [1, 1024, 16, 64]);  mul_219 = None
    permute_346: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None
    clone_248: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_651: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_248, [16, -1, 64]);  clone_248 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_3: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_651, 0);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_644: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_342: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg476_1, [1, 0]);  arg476_1 = None
    addmm_177: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg477_1, view_644, permute_342);  arg477_1 = view_644 = permute_342 = None
    view_645: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_177, [1, 1024, 1024]);  addmm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_646: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_645, [1, -1, 16, 64]);  view_645 = None
    permute_343: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_646, [0, 2, 1, 3]);  view_646 = None
    clone_246: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_652: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_246, [16, -1, 64]);  clone_246 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_4: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_652, 0);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_647: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_344: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg478_1, [1, 0]);  arg478_1 = None
    addmm_178: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg479_1, view_647, permute_344);  arg479_1 = view_647 = permute_344 = None
    view_648: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_178, [1, 1024, 1024]);  addmm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_649: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_648, [1, -1, 16, 64]);  view_648 = None
    permute_345: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
    clone_247: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_653: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_247, [16, -1, 64]);  clone_247 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_5: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_653, 0);  view_653 = None
    _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
    getitem_129: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default_1[0];  _scaled_dot_product_flash_attention_default_1 = None
    squeeze_dim_1: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_129, 0);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_654: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_1, [1, 16, 1024, 64]);  squeeze_dim_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_348: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_250: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_655: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_250, [1, 1024, 1024]);  clone_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_656: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_655, [1024, 1024]);  view_655 = None
    permute_349: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg480_1, [1, 0]);  arg480_1 = None
    addmm_179: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg481_1, view_656, permute_349);  arg481_1 = view_656 = permute_349 = None
    view_657: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_179, [1, 1024, 1024]);  addmm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_211: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_208, view_657);  add_208 = view_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_59 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 1024, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 1024, 1]" = var_mean_59[1];  var_mean_59 = None
    sub_94: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_211, getitem_119);  getitem_119 = None
    add_212: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    mul_220: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_59);  sub_94 = rsqrt_59 = None
    mul_221: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_220, arg482_1);  mul_220 = arg482_1 = None
    add_213: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_221, arg483_1);  mul_221 = arg483_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_658: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_213, [1024, 1024]);  add_213 = None
    permute_350: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg484_1, [1, 0]);  arg484_1 = None
    addmm_180: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg485_1, view_658, permute_350);  arg485_1 = view_658 = permute_350 = None
    view_659: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_180, [1, 1024, 4096]);  addmm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_222: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_659, 0.5)
    mul_223: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_659, 0.7071067811865476);  view_659 = None
    erf_22: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_223);  mul_223 = None
    add_214: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_224: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_222, add_214);  mul_222 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_660: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_224, [1024, 4096]);  mul_224 = None
    permute_351: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg486_1, [1, 0]);  arg486_1 = None
    addmm_181: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg487_1, view_660, permute_351);  arg487_1 = view_660 = permute_351 = None
    view_661: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_181, [1, 1024, 1024]);  addmm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_215: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_211, view_661);  add_211 = view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_60 = torch.ops.aten.var_mean.correction(add_215, [2], correction = 0, keepdim = True)
    getitem_120: "f32[1, 1024, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 1024, 1]" = var_mean_60[1];  var_mean_60 = None
    sub_95: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_215, getitem_121);  getitem_121 = None
    add_216: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_60: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    mul_225: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_60);  sub_95 = rsqrt_60 = None
    mul_226: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_225, arg488_1);  mul_225 = arg488_1 = None
    add_217: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_226, arg489_1);  mul_226 = arg489_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_662: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_217, [1024, 1024])
    permute_352: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg490_1, [1, 0]);  arg490_1 = None
    addmm_182: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg491_1, view_662, permute_352);  arg491_1 = view_662 = permute_352 = None
    view_663: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_182, [1, 1024, 1024]);  addmm_182 = None
    mul_227: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_663, 0.125);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_670: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_227, [1, 1024, 16, 64]);  mul_227 = None
    permute_357: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None
    clone_256: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_671: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_256, [16, -1, 64]);  clone_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_664: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_217, [1024, 1024])
    permute_353: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg492_1, [1, 0]);  arg492_1 = None
    addmm_183: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg493_1, view_664, permute_353);  arg493_1 = view_664 = permute_353 = None
    view_665: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_183, [1, 1024, 1024]);  addmm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_666: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_665, [1, -1, 16, 64]);  view_665 = None
    permute_354: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_666, [0, 2, 1, 3]);  view_666 = None
    clone_254: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_672: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_254, [16, -1, 64]);  clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_358: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_672, [0, 2, 1]);  view_672 = None
    bmm_68: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_671, permute_358);  view_671 = permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_674: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_68, [1, 16, 1024, 1024]);  bmm_68 = None
    add_218: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_674, expand_3);  view_674 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_675: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_218, [16, 1024, 1024]);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_34: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_675, [-1], True)
    sub_96: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_675, amax_34);  view_675 = amax_34 = None
    exp_34: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_96);  sub_96 = None
    sum_36: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
    div_34: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_34, sum_36);  exp_34 = sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_667: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_217, [1024, 1024]);  add_217 = None
    permute_355: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg494_1, [1, 0]);  arg494_1 = None
    addmm_184: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg495_1, view_667, permute_355);  arg495_1 = view_667 = permute_355 = None
    view_668: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_184, [1, 1024, 1024]);  addmm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_669: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_668, [1, -1, 16, 64]);  view_668 = None
    permute_356: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_669, [0, 2, 1, 3]);  view_669 = None
    clone_255: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_673: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_255, [16, -1, 64]);  clone_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_69: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_34, view_673);  div_34 = view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_676: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_69, [1, 16, 1024, 64]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_359: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_258: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_677: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_258, [1, 1024, 1024]);  clone_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_678: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_677, [1024, 1024]);  view_677 = None
    permute_360: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg496_1, [1, 0]);  arg496_1 = None
    addmm_185: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg497_1, view_678, permute_360);  arg497_1 = view_678 = permute_360 = None
    view_679: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_185, [1, 1024, 1024]);  addmm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_219: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_215, view_679);  add_215 = view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_61 = torch.ops.aten.var_mean.correction(add_219, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 1024, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 1024, 1]" = var_mean_61[1];  var_mean_61 = None
    sub_97: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_219, getitem_123);  getitem_123 = None
    add_220: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_61: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    mul_228: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_61);  sub_97 = rsqrt_61 = None
    mul_229: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_228, arg498_1);  mul_228 = arg498_1 = None
    add_221: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_229, arg499_1);  mul_229 = arg499_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_680: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_221, [1024, 1024]);  add_221 = None
    permute_361: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg500_1, [1, 0]);  arg500_1 = None
    addmm_186: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg501_1, view_680, permute_361);  arg501_1 = view_680 = permute_361 = None
    view_681: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_186, [1, 1024, 1024]);  addmm_186 = None
    mul_230: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_681, 0.125);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_688: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_230, [1, 1024, 16, 64]);  mul_230 = None
    permute_366: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
    clone_262: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_689: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_262, [16, -1, 64]);  clone_262 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_689, 0);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_682: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_362: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg502_1, [1, 0]);  arg502_1 = None
    addmm_187: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg503_1, view_682, permute_362);  arg503_1 = view_682 = permute_362 = None
    view_683: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_187, [1, 1024, 1024]);  addmm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_684: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_683, [1, -1, 16, 64]);  view_683 = None
    permute_363: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_684, [0, 2, 1, 3]);  view_684 = None
    clone_260: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_690: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_260, [16, -1, 64]);  clone_260 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_1: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_690, 0);  view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_685: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_89, [1024, 1024])
    permute_364: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
    addmm_188: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg505_1, view_685, permute_364);  arg505_1 = view_685 = permute_364 = None
    view_686: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_188, [1, 1024, 1024]);  addmm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_687: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_686, [1, -1, 16, 64]);  view_686 = None
    permute_365: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_687, [0, 2, 1, 3]);  view_687 = None
    clone_261: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_691: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_261, [16, -1, 64]);  clone_261 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_691, 0);  view_691 = None
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem_128: "f32[1, 16, 1024, 64]" = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
    squeeze_dim: "f32[16, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_128, 0);  getitem_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_692: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim, [1, 16, 1024, 64]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_368: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_264: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_693: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_264, [1, 1024, 1024]);  clone_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_694: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_693, [1024, 1024]);  view_693 = None
    permute_369: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg506_1, [1, 0]);  arg506_1 = None
    addmm_189: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg507_1, view_694, permute_369);  arg507_1 = view_694 = permute_369 = None
    view_695: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_189, [1, 1024, 1024]);  addmm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    add_222: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_219, view_695);  add_219 = view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_62 = torch.ops.aten.var_mean.correction(add_222, [2], correction = 0, keepdim = True)
    getitem_124: "f32[1, 1024, 1]" = var_mean_62[0]
    getitem_125: "f32[1, 1024, 1]" = var_mean_62[1];  var_mean_62 = None
    sub_99: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_222, getitem_125);  getitem_125 = None
    add_223: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_62: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    mul_231: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_62);  sub_99 = rsqrt_62 = None
    mul_232: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_231, arg508_1);  mul_231 = arg508_1 = None
    add_224: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_232, arg509_1);  mul_232 = arg509_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_696: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_224, [1024, 1024]);  add_224 = None
    permute_370: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
    addmm_190: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg511_1, view_696, permute_370);  arg511_1 = view_696 = permute_370 = None
    view_697: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_190, [1, 1024, 4096]);  addmm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_233: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_697, 0.5)
    mul_234: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_697, 0.7071067811865476);  view_697 = None
    erf_23: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
    add_225: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_235: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_233, add_225);  mul_233 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_698: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_235, [1024, 4096]);  mul_235 = None
    permute_371: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg512_1, [1, 0]);  arg512_1 = None
    addmm_191: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg513_1, view_698, permute_371);  arg513_1 = view_698 = permute_371 = None
    view_699: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_191, [1, 1024, 1024]);  addmm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_226: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_222, view_699);  add_222 = view_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1132, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_63 = torch.ops.aten.var_mean.correction(add_226, [2], correction = 0, keepdim = True)
    getitem_126: "f32[1, 1024, 1]" = var_mean_63[0]
    getitem_127: "f32[1, 1024, 1]" = var_mean_63[1];  var_mean_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1379, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_703: "i64[1024]" = torch.ops.aten.reshape.default(arg518_1, [-1]);  arg518_1 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_703, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1132, code: hidden_states = self.layer_norm(hidden_states)
    sub_100: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_226, getitem_127);  add_226 = getitem_127 = None
    add_227: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_63: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    mul_236: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_63);  sub_100 = rsqrt_63 = None
    mul_237: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_236, arg514_1);  mul_236 = arg514_1 = None
    add_228: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_237, arg515_1);  mul_237 = arg515_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1374, code: lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    view_700: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_228, [1024, 1024]);  add_228 = None
    permute_372: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg516_1, [1, 0]);  arg516_1 = None
    mm: "f32[1024, 50265]" = torch.ops.aten.mm.default(view_700, permute_372);  view_700 = permute_372 = None
    view_701: "f32[1, 1024, 50265]" = torch.ops.aten.reshape.default(mm, [1, 1024, 50265]);  mm = None
    add_229: "f32[1, 1024, 50265]" = torch.ops.aten.add.Tensor(view_701, arg517_1);  view_701 = arg517_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1379, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_702: "f32[1024, 50265]" = torch.ops.aten.reshape.default(add_229, [-1, 50265])
    amax_36: "f32[1024, 1]" = torch.ops.aten.amax.default(view_702, [1], True)
    sub_101: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(view_702, amax_36);  view_702 = amax_36 = None
    exp_36: "f32[1024, 50265]" = torch.ops.aten.exp.default(sub_101)
    sum_38: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [1], True);  exp_36 = None
    log: "f32[1024, 1]" = torch.ops.aten.log.default(sum_38);  sum_38 = None
    sub_102: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(sub_101, log);  sub_101 = log = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_703, -100)
    full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_2: "i64[1024]" = torch.ops.aten.where.self(ne_1, view_703, full_default_3);  ne_1 = full_default_3 = None
    unsqueeze_5: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_102, 1, unsqueeze_5);  sub_102 = unsqueeze_5 = None
    squeeze_1: "f32[1024]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg: "f32[1024]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
    full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_3: "f32[1024]" = torch.ops.aten.where.self(ne_2, neg, full_default_4);  ne_2 = neg = full_default_4 = None
    sum_40: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    ne_3: "b8[1024]" = torch.ops.aten.ne.Scalar(view_703, -100);  view_703 = None
    sum_39: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_39, torch.float32);  sum_39 = None
    div_36: "f32[]" = torch.ops.aten.div.Tensor(sum_40, convert_element_type);  sum_40 = convert_element_type = None
    return (div_36, add_229, add_89)
    