from __future__ import annotations



def forward(self, arg0_1: "f32[128112, 1024]", arg1_1: "f32[1024]", arg2_1: "f32[1024]", arg3_1: "f32[1024, 1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024, 1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024]", arg12_1: "f32[1024]", arg13_1: "f32[4096, 1024]", arg14_1: "f32[4096]", arg15_1: "f32[1024, 4096]", arg16_1: "f32[1024]", arg17_1: "f32[1024]", arg18_1: "f32[1024]", arg19_1: "f32[1024, 1024]", arg20_1: "f32[1024]", arg21_1: "f32[1024, 1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024]", arg28_1: "f32[1024]", arg29_1: "f32[4096, 1024]", arg30_1: "f32[4096]", arg31_1: "f32[1024, 4096]", arg32_1: "f32[1024]", arg33_1: "f32[1024]", arg34_1: "f32[1024]", arg35_1: "f32[1024, 1024]", arg36_1: "f32[1024]", arg37_1: "f32[1024, 1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024]", arg44_1: "f32[1024]", arg45_1: "f32[4096, 1024]", arg46_1: "f32[4096]", arg47_1: "f32[1024, 4096]", arg48_1: "f32[1024]", arg49_1: "f32[1024]", arg50_1: "f32[1024]", arg51_1: "f32[1024, 1024]", arg52_1: "f32[1024]", arg53_1: "f32[1024, 1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024]", arg60_1: "f32[1024]", arg61_1: "f32[4096, 1024]", arg62_1: "f32[4096]", arg63_1: "f32[1024, 4096]", arg64_1: "f32[1024]", arg65_1: "f32[1024]", arg66_1: "f32[1024]", arg67_1: "f32[1024, 1024]", arg68_1: "f32[1024]", arg69_1: "f32[1024, 1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024]", arg76_1: "f32[1024]", arg77_1: "f32[4096, 1024]", arg78_1: "f32[4096]", arg79_1: "f32[1024, 4096]", arg80_1: "f32[1024]", arg81_1: "f32[1024]", arg82_1: "f32[1024]", arg83_1: "f32[1024, 1024]", arg84_1: "f32[1024]", arg85_1: "f32[1024, 1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024]", arg92_1: "f32[1024]", arg93_1: "f32[4096, 1024]", arg94_1: "f32[4096]", arg95_1: "f32[1024, 4096]", arg96_1: "f32[1024]", arg97_1: "f32[1024]", arg98_1: "f32[1024]", arg99_1: "f32[1024, 1024]", arg100_1: "f32[1024]", arg101_1: "f32[1024, 1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024]", arg108_1: "f32[1024]", arg109_1: "f32[4096, 1024]", arg110_1: "f32[4096]", arg111_1: "f32[1024, 4096]", arg112_1: "f32[1024]", arg113_1: "f32[1024]", arg114_1: "f32[1024]", arg115_1: "f32[1024, 1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024, 1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024]", arg124_1: "f32[1024]", arg125_1: "f32[4096, 1024]", arg126_1: "f32[4096]", arg127_1: "f32[1024, 4096]", arg128_1: "f32[1024]", arg129_1: "f32[1024]", arg130_1: "f32[1024]", arg131_1: "f32[1024, 1024]", arg132_1: "f32[1024]", arg133_1: "f32[1024, 1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024]", arg140_1: "f32[1024]", arg141_1: "f32[4096, 1024]", arg142_1: "f32[4096]", arg143_1: "f32[1024, 4096]", arg144_1: "f32[1024]", arg145_1: "f32[1024]", arg146_1: "f32[1024]", arg147_1: "f32[1024, 1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024, 1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024]", arg156_1: "f32[1024]", arg157_1: "f32[4096, 1024]", arg158_1: "f32[4096]", arg159_1: "f32[1024, 4096]", arg160_1: "f32[1024]", arg161_1: "f32[1024]", arg162_1: "f32[1024]", arg163_1: "f32[1024, 1024]", arg164_1: "f32[1024]", arg165_1: "f32[1024, 1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024]", arg172_1: "f32[1024]", arg173_1: "f32[4096, 1024]", arg174_1: "f32[4096]", arg175_1: "f32[1024, 4096]", arg176_1: "f32[1024]", arg177_1: "f32[1024]", arg178_1: "f32[1024]", arg179_1: "f32[1024, 1024]", arg180_1: "f32[1024]", arg181_1: "f32[1024, 1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024]", arg188_1: "f32[1024]", arg189_1: "f32[4096, 1024]", arg190_1: "f32[4096]", arg191_1: "f32[1024, 4096]", arg192_1: "f32[1024]", arg193_1: "f32[1024]", arg194_1: "f32[1024]", arg195_1: "f32[128112, 1024]", arg196_1: "f32[1024]", arg197_1: "f32[1024]", arg198_1: "f32[1024, 1024]", arg199_1: "f32[1024]", arg200_1: "f32[1024, 1024]", arg201_1: "f32[1024]", arg202_1: "f32[1024, 1024]", arg203_1: "f32[1024]", arg204_1: "f32[1024, 1024]", arg205_1: "f32[1024]", arg206_1: "f32[1024]", arg207_1: "f32[1024]", arg208_1: "f32[1024, 1024]", arg209_1: "f32[1024]", arg210_1: "f32[1024, 1024]", arg211_1: "f32[1024]", arg212_1: "f32[1024, 1024]", arg213_1: "f32[1024]", arg214_1: "f32[1024, 1024]", arg215_1: "f32[1024]", arg216_1: "f32[1024]", arg217_1: "f32[1024]", arg218_1: "f32[4096, 1024]", arg219_1: "f32[4096]", arg220_1: "f32[1024, 4096]", arg221_1: "f32[1024]", arg222_1: "f32[1024]", arg223_1: "f32[1024]", arg224_1: "f32[1024, 1024]", arg225_1: "f32[1024]", arg226_1: "f32[1024, 1024]", arg227_1: "f32[1024]", arg228_1: "f32[1024, 1024]", arg229_1: "f32[1024]", arg230_1: "f32[1024, 1024]", arg231_1: "f32[1024]", arg232_1: "f32[1024]", arg233_1: "f32[1024]", arg234_1: "f32[1024, 1024]", arg235_1: "f32[1024]", arg236_1: "f32[1024, 1024]", arg237_1: "f32[1024]", arg238_1: "f32[1024, 1024]", arg239_1: "f32[1024]", arg240_1: "f32[1024, 1024]", arg241_1: "f32[1024]", arg242_1: "f32[1024]", arg243_1: "f32[1024]", arg244_1: "f32[4096, 1024]", arg245_1: "f32[4096]", arg246_1: "f32[1024, 4096]", arg247_1: "f32[1024]", arg248_1: "f32[1024]", arg249_1: "f32[1024]", arg250_1: "f32[1024, 1024]", arg251_1: "f32[1024]", arg252_1: "f32[1024, 1024]", arg253_1: "f32[1024]", arg254_1: "f32[1024, 1024]", arg255_1: "f32[1024]", arg256_1: "f32[1024, 1024]", arg257_1: "f32[1024]", arg258_1: "f32[1024]", arg259_1: "f32[1024]", arg260_1: "f32[1024, 1024]", arg261_1: "f32[1024]", arg262_1: "f32[1024, 1024]", arg263_1: "f32[1024]", arg264_1: "f32[1024, 1024]", arg265_1: "f32[1024]", arg266_1: "f32[1024, 1024]", arg267_1: "f32[1024]", arg268_1: "f32[1024]", arg269_1: "f32[1024]", arg270_1: "f32[4096, 1024]", arg271_1: "f32[4096]", arg272_1: "f32[1024, 4096]", arg273_1: "f32[1024]", arg274_1: "f32[1024]", arg275_1: "f32[1024]", arg276_1: "f32[1024, 1024]", arg277_1: "f32[1024]", arg278_1: "f32[1024, 1024]", arg279_1: "f32[1024]", arg280_1: "f32[1024, 1024]", arg281_1: "f32[1024]", arg282_1: "f32[1024, 1024]", arg283_1: "f32[1024]", arg284_1: "f32[1024]", arg285_1: "f32[1024]", arg286_1: "f32[1024, 1024]", arg287_1: "f32[1024]", arg288_1: "f32[1024, 1024]", arg289_1: "f32[1024]", arg290_1: "f32[1024, 1024]", arg291_1: "f32[1024]", arg292_1: "f32[1024, 1024]", arg293_1: "f32[1024]", arg294_1: "f32[1024]", arg295_1: "f32[1024]", arg296_1: "f32[4096, 1024]", arg297_1: "f32[4096]", arg298_1: "f32[1024, 4096]", arg299_1: "f32[1024]", arg300_1: "f32[1024]", arg301_1: "f32[1024]", arg302_1: "f32[1024, 1024]", arg303_1: "f32[1024]", arg304_1: "f32[1024, 1024]", arg305_1: "f32[1024]", arg306_1: "f32[1024, 1024]", arg307_1: "f32[1024]", arg308_1: "f32[1024, 1024]", arg309_1: "f32[1024]", arg310_1: "f32[1024]", arg311_1: "f32[1024]", arg312_1: "f32[1024, 1024]", arg313_1: "f32[1024]", arg314_1: "f32[1024, 1024]", arg315_1: "f32[1024]", arg316_1: "f32[1024, 1024]", arg317_1: "f32[1024]", arg318_1: "f32[1024, 1024]", arg319_1: "f32[1024]", arg320_1: "f32[1024]", arg321_1: "f32[1024]", arg322_1: "f32[4096, 1024]", arg323_1: "f32[4096]", arg324_1: "f32[1024, 4096]", arg325_1: "f32[1024]", arg326_1: "f32[1024]", arg327_1: "f32[1024]", arg328_1: "f32[1024, 1024]", arg329_1: "f32[1024]", arg330_1: "f32[1024, 1024]", arg331_1: "f32[1024]", arg332_1: "f32[1024, 1024]", arg333_1: "f32[1024]", arg334_1: "f32[1024, 1024]", arg335_1: "f32[1024]", arg336_1: "f32[1024]", arg337_1: "f32[1024]", arg338_1: "f32[1024, 1024]", arg339_1: "f32[1024]", arg340_1: "f32[1024, 1024]", arg341_1: "f32[1024]", arg342_1: "f32[1024, 1024]", arg343_1: "f32[1024]", arg344_1: "f32[1024, 1024]", arg345_1: "f32[1024]", arg346_1: "f32[1024]", arg347_1: "f32[1024]", arg348_1: "f32[4096, 1024]", arg349_1: "f32[4096]", arg350_1: "f32[1024, 4096]", arg351_1: "f32[1024]", arg352_1: "f32[1024]", arg353_1: "f32[1024]", arg354_1: "f32[1024, 1024]", arg355_1: "f32[1024]", arg356_1: "f32[1024, 1024]", arg357_1: "f32[1024]", arg358_1: "f32[1024, 1024]", arg359_1: "f32[1024]", arg360_1: "f32[1024, 1024]", arg361_1: "f32[1024]", arg362_1: "f32[1024]", arg363_1: "f32[1024]", arg364_1: "f32[1024, 1024]", arg365_1: "f32[1024]", arg366_1: "f32[1024, 1024]", arg367_1: "f32[1024]", arg368_1: "f32[1024, 1024]", arg369_1: "f32[1024]", arg370_1: "f32[1024, 1024]", arg371_1: "f32[1024]", arg372_1: "f32[1024]", arg373_1: "f32[1024]", arg374_1: "f32[4096, 1024]", arg375_1: "f32[4096]", arg376_1: "f32[1024, 4096]", arg377_1: "f32[1024]", arg378_1: "f32[1024]", arg379_1: "f32[1024]", arg380_1: "f32[1024, 1024]", arg381_1: "f32[1024]", arg382_1: "f32[1024, 1024]", arg383_1: "f32[1024]", arg384_1: "f32[1024, 1024]", arg385_1: "f32[1024]", arg386_1: "f32[1024, 1024]", arg387_1: "f32[1024]", arg388_1: "f32[1024]", arg389_1: "f32[1024]", arg390_1: "f32[1024, 1024]", arg391_1: "f32[1024]", arg392_1: "f32[1024, 1024]", arg393_1: "f32[1024]", arg394_1: "f32[1024, 1024]", arg395_1: "f32[1024]", arg396_1: "f32[1024, 1024]", arg397_1: "f32[1024]", arg398_1: "f32[1024]", arg399_1: "f32[1024]", arg400_1: "f32[4096, 1024]", arg401_1: "f32[4096]", arg402_1: "f32[1024, 4096]", arg403_1: "f32[1024]", arg404_1: "f32[1024]", arg405_1: "f32[1024]", arg406_1: "f32[1024, 1024]", arg407_1: "f32[1024]", arg408_1: "f32[1024, 1024]", arg409_1: "f32[1024]", arg410_1: "f32[1024, 1024]", arg411_1: "f32[1024]", arg412_1: "f32[1024, 1024]", arg413_1: "f32[1024]", arg414_1: "f32[1024]", arg415_1: "f32[1024]", arg416_1: "f32[1024, 1024]", arg417_1: "f32[1024]", arg418_1: "f32[1024, 1024]", arg419_1: "f32[1024]", arg420_1: "f32[1024, 1024]", arg421_1: "f32[1024]", arg422_1: "f32[1024, 1024]", arg423_1: "f32[1024]", arg424_1: "f32[1024]", arg425_1: "f32[1024]", arg426_1: "f32[4096, 1024]", arg427_1: "f32[4096]", arg428_1: "f32[1024, 4096]", arg429_1: "f32[1024]", arg430_1: "f32[1024]", arg431_1: "f32[1024]", arg432_1: "f32[1024, 1024]", arg433_1: "f32[1024]", arg434_1: "f32[1024, 1024]", arg435_1: "f32[1024]", arg436_1: "f32[1024, 1024]", arg437_1: "f32[1024]", arg438_1: "f32[1024, 1024]", arg439_1: "f32[1024]", arg440_1: "f32[1024]", arg441_1: "f32[1024]", arg442_1: "f32[1024, 1024]", arg443_1: "f32[1024]", arg444_1: "f32[1024, 1024]", arg445_1: "f32[1024]", arg446_1: "f32[1024, 1024]", arg447_1: "f32[1024]", arg448_1: "f32[1024, 1024]", arg449_1: "f32[1024]", arg450_1: "f32[1024]", arg451_1: "f32[1024]", arg452_1: "f32[4096, 1024]", arg453_1: "f32[4096]", arg454_1: "f32[1024, 4096]", arg455_1: "f32[1024]", arg456_1: "f32[1024]", arg457_1: "f32[1024]", arg458_1: "f32[1024, 1024]", arg459_1: "f32[1024]", arg460_1: "f32[1024, 1024]", arg461_1: "f32[1024]", arg462_1: "f32[1024, 1024]", arg463_1: "f32[1024]", arg464_1: "f32[1024, 1024]", arg465_1: "f32[1024]", arg466_1: "f32[1024]", arg467_1: "f32[1024]", arg468_1: "f32[1024, 1024]", arg469_1: "f32[1024]", arg470_1: "f32[1024, 1024]", arg471_1: "f32[1024]", arg472_1: "f32[1024, 1024]", arg473_1: "f32[1024]", arg474_1: "f32[1024, 1024]", arg475_1: "f32[1024]", arg476_1: "f32[1024]", arg477_1: "f32[1024]", arg478_1: "f32[4096, 1024]", arg479_1: "f32[4096]", arg480_1: "f32[1024, 4096]", arg481_1: "f32[1024]", arg482_1: "f32[1024]", arg483_1: "f32[1024]", arg484_1: "f32[1024, 1024]", arg485_1: "f32[1024]", arg486_1: "f32[1024, 1024]", arg487_1: "f32[1024]", arg488_1: "f32[1024, 1024]", arg489_1: "f32[1024]", arg490_1: "f32[1024, 1024]", arg491_1: "f32[1024]", arg492_1: "f32[1024]", arg493_1: "f32[1024]", arg494_1: "f32[1024, 1024]", arg495_1: "f32[1024]", arg496_1: "f32[1024, 1024]", arg497_1: "f32[1024]", arg498_1: "f32[1024, 1024]", arg499_1: "f32[1024]", arg500_1: "f32[1024, 1024]", arg501_1: "f32[1024]", arg502_1: "f32[1024]", arg503_1: "f32[1024]", arg504_1: "f32[4096, 1024]", arg505_1: "f32[4096]", arg506_1: "f32[1024, 4096]", arg507_1: "f32[1024]", arg508_1: "f32[1024]", arg509_1: "f32[1024]", arg510_1: "f32[128112, 1024]", arg511_1: "f32[1026, 1024]", arg512_1: "f32[1026, 1024]", arg513_1: "i64[1, 128]", arg514_1: "i64[1, 128]", arg515_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:779, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.view.default(arg515_1, [-1, 128]);  arg515_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:786, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 128, 1024]" = torch.ops.aten.embedding.default(arg0_1, view, 1);  arg0_1 = None
    mul: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:113, code: mask = input_ids.ne(padding_idx).int()
    ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(view, 1);  view = None
    convert_element_type: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:114, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum: "i64[1, 128]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    convert_element_type_1: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
    add: "i32[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
    mul_1: "i32[1, 128]" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:115, code: return incremental_indices.long() + padding_idx
    convert_element_type_2: "i64[1, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
    add_1: "i64[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:176, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_1: "i64[128]" = torch.ops.aten.view.default(add_1, [-1]);  add_1 = None
    index: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg511_1, [view_1]);  arg511_1 = view_1 = None
    view_2: "f32[1, 128, 1024]" = torch.ops.aten.view.default(index, [1, 128, 1024]);  index = None
    alias: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(view_2);  view_2 = None
    alias_1: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:791, code: hidden_states = inputs_embeds + embed_pos
    add_2: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul, alias_1);  mul = alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:792, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  getitem_1 = None
    mul_2: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_3: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg1_1);  mul_2 = arg1_1 = None
    add_4: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_3, arg2_1);  mul_3 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_3: "f32[128, 1024]" = torch.ops.aten.view.default(add_4, [128, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
    addmm: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg4_1, view_3, permute);  arg4_1 = view_3 = permute = None
    view_4: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm, [1, 128, 1024]);  addmm = None
    mul_4: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_4, 0.125);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_5: "f32[128, 1024]" = torch.ops.aten.view.default(add_4, [128, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm_1: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg6_1, view_5, permute_1);  arg6_1 = view_5 = permute_1 = None
    view_6: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 128, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_6, [1, -1, 16, 64]);  view_6 = None
    permute_2: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_8: "f32[128, 1024]" = torch.ops.aten.view.default(add_4, [128, 1024]);  add_4 = None
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm_2: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg8_1, view_8, permute_3);  arg8_1 = view_8 = permute_3 = None
    view_9: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_2, [1, 128, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_9, [1, -1, 16, 64]);  view_9 = None
    permute_4: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_2: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_11: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_4, [1, 128, 16, 64]);  mul_4 = None
    permute_5: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_3: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_12: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_3, [16, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_13: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_1, [16, -1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_14: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_2, [16, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_12, permute_6);  view_12 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm, [-1], True)
    sub_1: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm, amax);  bmm = amax = None
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_4: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_4, view_14);  clone_4 = view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_15: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, [1, 16, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_16: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_5, [1, 128, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_17: "f32[128, 1024]" = torch.ops.aten.view.default(view_16, [128, 1024]);  view_16 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_3: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_17, permute_8);  arg10_1 = view_17 = permute_8 = None
    view_18: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_3, [1, 128, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_6: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_5: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
    mul_5: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_6: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_5, arg11_1);  mul_5 = arg11_1 = None
    add_7: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_6, arg12_1);  mul_6 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_19: "f32[128, 1024]" = torch.ops.aten.view.default(add_7, [128, 1024]);  add_7 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    addmm_4: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg14_1, view_19, permute_9);  arg14_1 = view_19 = permute_9 = None
    view_20: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_4, [1, 128, 4096]);  addmm_4 = None
    relu: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_7: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu);  relu = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_21: "f32[128, 4096]" = torch.ops.aten.view.default(clone_7, [128, 4096]);  clone_7 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    addmm_5: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg16_1, view_21, permute_10);  arg16_1 = view_21 = permute_10 = None
    view_22: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 128, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_8: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_22);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_8: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_5, clone_8);  add_5 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  getitem_5 = None
    mul_7: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_8: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, arg17_1);  mul_7 = arg17_1 = None
    add_10: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_8, arg18_1);  mul_8 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_23: "f32[128, 1024]" = torch.ops.aten.view.default(add_10, [128, 1024])
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_6: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg20_1, view_23, permute_11);  arg20_1 = view_23 = permute_11 = None
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_6, [1, 128, 1024]);  addmm_6 = None
    mul_9: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_24, 0.125);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_25: "f32[128, 1024]" = torch.ops.aten.view.default(add_10, [128, 1024])
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    addmm_7: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg22_1, view_25, permute_12);  arg22_1 = view_25 = permute_12 = None
    view_26: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_7, [1, 128, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_27: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_26, [1, -1, 16, 64]);  view_26 = None
    permute_13: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    clone_9: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_28: "f32[128, 1024]" = torch.ops.aten.view.default(add_10, [128, 1024]);  add_10 = None
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_8: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_28, permute_14);  arg24_1 = view_28 = permute_14 = None
    view_29: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_8, [1, 128, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_30: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_29, [1, -1, 16, 64]);  view_29 = None
    permute_15: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_10: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_9, [1, 128, 16, 64]);  mul_9 = None
    permute_16: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_11: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_32: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_11, [16, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_33: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_9, [16, -1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_34: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_10, [16, -1, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_32, permute_17);  view_32 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_2, [-1], True)
    sub_4: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1);  bmm_2 = amax_1 = None
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_12: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_12, view_34);  clone_12 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_35: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_3, [1, 16, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_36: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_13, [1, 128, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_37: "f32[128, 1024]" = torch.ops.aten.view.default(view_36, [128, 1024]);  view_36 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_9: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_37, permute_19);  arg26_1 = view_37 = permute_19 = None
    view_38: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_9, [1, 128, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_14: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_38);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_11: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_8, clone_14);  add_8 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  getitem_7 = None
    mul_10: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_11: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_10, arg27_1);  mul_10 = arg27_1 = None
    add_13: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_11, arg28_1);  mul_11 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_39: "f32[128, 1024]" = torch.ops.aten.view.default(add_13, [128, 1024]);  add_13 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    addmm_10: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg30_1, view_39, permute_20);  arg30_1 = view_39 = permute_20 = None
    view_40: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_10, [1, 128, 4096]);  addmm_10 = None
    relu_1: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_15: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_1);  relu_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_41: "f32[128, 4096]" = torch.ops.aten.view.default(clone_15, [128, 4096]);  clone_15 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    addmm_11: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg32_1, view_41, permute_21);  arg32_1 = view_41 = permute_21 = None
    view_42: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 128, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_16: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_42);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_11, clone_16);  add_11 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  getitem_9 = None
    mul_12: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_13: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, arg33_1);  mul_12 = arg33_1 = None
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_13, arg34_1);  mul_13 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_43: "f32[128, 1024]" = torch.ops.aten.view.default(add_16, [128, 1024])
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    addmm_12: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg36_1, view_43, permute_22);  arg36_1 = view_43 = permute_22 = None
    view_44: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_12, [1, 128, 1024]);  addmm_12 = None
    mul_14: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_44, 0.125);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_45: "f32[128, 1024]" = torch.ops.aten.view.default(add_16, [128, 1024])
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    addmm_13: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg38_1, view_45, permute_23);  arg38_1 = view_45 = permute_23 = None
    view_46: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_13, [1, 128, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_47: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_46, [1, -1, 16, 64]);  view_46 = None
    permute_24: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    clone_17: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_48: "f32[128, 1024]" = torch.ops.aten.view.default(add_16, [128, 1024]);  add_16 = None
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_14: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg40_1, view_48, permute_25);  arg40_1 = view_48 = permute_25 = None
    view_49: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_14, [1, 128, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_49, [1, -1, 16, 64]);  view_49 = None
    permute_26: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_18: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_51: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_14, [1, 128, 16, 64]);  mul_14 = None
    permute_27: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    clone_19: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_52: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_19, [16, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_53: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_17, [16, -1, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_54: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_18, [16, -1, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_52, permute_28);  view_52 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_4, [-1], True)
    sub_7: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_4, amax_2);  bmm_4 = amax_2 = None
    exp_2: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_20: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_20, view_54);  clone_20 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_55: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_5, [1, 16, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_56: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_21, [1, 128, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_57: "f32[128, 1024]" = torch.ops.aten.view.default(view_56, [128, 1024]);  view_56 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_15: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_57, permute_30);  arg42_1 = view_57 = permute_30 = None
    view_58: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_15, [1, 128, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_22: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_58);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_17: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_14, clone_22);  add_14 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_8: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  getitem_11 = None
    mul_15: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_16: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, arg43_1);  mul_15 = arg43_1 = None
    add_19: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_16, arg44_1);  mul_16 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_59: "f32[128, 1024]" = torch.ops.aten.view.default(add_19, [128, 1024]);  add_19 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    addmm_16: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg46_1, view_59, permute_31);  arg46_1 = view_59 = permute_31 = None
    view_60: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_16, [1, 128, 4096]);  addmm_16 = None
    relu_2: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_23: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_2);  relu_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_61: "f32[128, 4096]" = torch.ops.aten.view.default(clone_23, [128, 4096]);  clone_23 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    addmm_17: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg48_1, view_61, permute_32);  arg48_1 = view_61 = permute_32 = None
    view_62: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 128, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_24: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_62);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_20: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_17, clone_24);  add_17 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_9: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_20, getitem_13);  getitem_13 = None
    mul_17: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_18: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg49_1);  mul_17 = arg49_1 = None
    add_22: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg50_1);  mul_18 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_63: "f32[128, 1024]" = torch.ops.aten.view.default(add_22, [128, 1024])
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    addmm_18: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg52_1, view_63, permute_33);  arg52_1 = view_63 = permute_33 = None
    view_64: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_18, [1, 128, 1024]);  addmm_18 = None
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_64, 0.125);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_65: "f32[128, 1024]" = torch.ops.aten.view.default(add_22, [128, 1024])
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    addmm_19: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg54_1, view_65, permute_34);  arg54_1 = view_65 = permute_34 = None
    view_66: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_19, [1, 128, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_67: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_66, [1, -1, 16, 64]);  view_66 = None
    permute_35: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
    clone_25: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_68: "f32[128, 1024]" = torch.ops.aten.view.default(add_22, [128, 1024]);  add_22 = None
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_20: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg56_1, view_68, permute_36);  arg56_1 = view_68 = permute_36 = None
    view_69: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_20, [1, 128, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_70: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_69, [1, -1, 16, 64]);  view_69 = None
    permute_37: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    clone_26: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_71: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_19, [1, 128, 16, 64]);  mul_19 = None
    permute_38: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_27: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_72: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_27, [16, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_73: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_25, [16, -1, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_74: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_26, [16, -1, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_72, permute_39);  view_72 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_6, [-1], True)
    sub_10: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_6, amax_3);  bmm_6 = amax_3 = None
    exp_3: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_28: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_28, view_74);  clone_28 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_75: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_7, [1, 16, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_76: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_29, [1, 128, 1024]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_77: "f32[128, 1024]" = torch.ops.aten.view.default(view_76, [128, 1024]);  view_76 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_21: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg58_1, view_77, permute_41);  arg58_1 = view_77 = permute_41 = None
    view_78: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_21, [1, 128, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_30: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_23: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_20, clone_30);  add_20 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_11: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_23, getitem_15);  getitem_15 = None
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_21: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, arg59_1);  mul_20 = arg59_1 = None
    add_25: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_21, arg60_1);  mul_21 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_79: "f32[128, 1024]" = torch.ops.aten.view.default(add_25, [128, 1024]);  add_25 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    addmm_22: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg62_1, view_79, permute_42);  arg62_1 = view_79 = permute_42 = None
    view_80: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_22, [1, 128, 4096]);  addmm_22 = None
    relu_3: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_31: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_3);  relu_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_81: "f32[128, 4096]" = torch.ops.aten.view.default(clone_31, [128, 4096]);  clone_31 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    addmm_23: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg64_1, view_81, permute_43);  arg64_1 = view_81 = permute_43 = None
    view_82: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 128, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_32: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_82);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_26: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_23, clone_32);  add_23 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_27: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_12: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_26, getitem_17);  getitem_17 = None
    mul_22: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_23: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg65_1);  mul_22 = arg65_1 = None
    add_28: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg66_1);  mul_23 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_83: "f32[128, 1024]" = torch.ops.aten.view.default(add_28, [128, 1024])
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    addmm_24: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg68_1, view_83, permute_44);  arg68_1 = view_83 = permute_44 = None
    view_84: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_24, [1, 128, 1024]);  addmm_24 = None
    mul_24: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_84, 0.125);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_85: "f32[128, 1024]" = torch.ops.aten.view.default(add_28, [128, 1024])
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    addmm_25: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg70_1, view_85, permute_45);  arg70_1 = view_85 = permute_45 = None
    view_86: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_25, [1, 128, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_87: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_86, [1, -1, 16, 64]);  view_86 = None
    permute_46: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
    clone_33: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_88: "f32[128, 1024]" = torch.ops.aten.view.default(add_28, [128, 1024]);  add_28 = None
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_26: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_88, permute_47);  arg72_1 = view_88 = permute_47 = None
    view_89: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_26, [1, 128, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_90: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_89, [1, -1, 16, 64]);  view_89 = None
    permute_48: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
    clone_34: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_91: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_24, [1, 128, 16, 64]);  mul_24 = None
    permute_49: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_91, [0, 2, 1, 3]);  view_91 = None
    clone_35: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_92: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_35, [16, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_93: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_33, [16, -1, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_94: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_34, [16, -1, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_92, permute_50);  view_92 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_8, [-1], True)
    sub_13: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_8, amax_4);  bmm_8 = amax_4 = None
    exp_4: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_36: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_36, view_94);  clone_36 = view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_95: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_9, [1, 16, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_96: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_37, [1, 128, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_97: "f32[128, 1024]" = torch.ops.aten.view.default(view_96, [128, 1024]);  view_96 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_27: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_97, permute_52);  arg74_1 = view_97 = permute_52 = None
    view_98: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_27, [1, 128, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_38: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_98);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_29: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_26, clone_38);  add_26 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_30: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_14: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_19);  getitem_19 = None
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_26: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, arg75_1);  mul_25 = arg75_1 = None
    add_31: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_26, arg76_1);  mul_26 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_99: "f32[128, 1024]" = torch.ops.aten.view.default(add_31, [128, 1024]);  add_31 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_28: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg78_1, view_99, permute_53);  arg78_1 = view_99 = permute_53 = None
    view_100: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_28, [1, 128, 4096]);  addmm_28 = None
    relu_4: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_39: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_4);  relu_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_101: "f32[128, 4096]" = torch.ops.aten.view.default(clone_39, [128, 4096]);  clone_39 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_29: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg80_1, view_101, permute_54);  arg80_1 = view_101 = permute_54 = None
    view_102: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 128, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_40: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_102);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_32: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_29, clone_40);  add_29 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_33: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_15: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_32, getitem_21);  getitem_21 = None
    mul_27: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_27, arg81_1);  mul_27 = arg81_1 = None
    add_34: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_28, arg82_1);  mul_28 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_103: "f32[128, 1024]" = torch.ops.aten.view.default(add_34, [128, 1024])
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    addmm_30: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg84_1, view_103, permute_55);  arg84_1 = view_103 = permute_55 = None
    view_104: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_30, [1, 128, 1024]);  addmm_30 = None
    mul_29: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_104, 0.125);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_105: "f32[128, 1024]" = torch.ops.aten.view.default(add_34, [128, 1024])
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_31: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg86_1, view_105, permute_56);  arg86_1 = view_105 = permute_56 = None
    view_106: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_31, [1, 128, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_107: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_106, [1, -1, 16, 64]);  view_106 = None
    permute_57: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    clone_41: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_108: "f32[128, 1024]" = torch.ops.aten.view.default(add_34, [128, 1024]);  add_34 = None
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_32: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg88_1, view_108, permute_58);  arg88_1 = view_108 = permute_58 = None
    view_109: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_32, [1, 128, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_110: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_109, [1, -1, 16, 64]);  view_109 = None
    permute_59: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    clone_42: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_111: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_29, [1, 128, 16, 64]);  mul_29 = None
    permute_60: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    clone_43: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_112: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_43, [16, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_113: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_41, [16, -1, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_114: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_42, [16, -1, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_112, permute_61);  view_112 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_10, [-1], True)
    sub_16: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_10, amax_5);  bmm_10 = amax_5 = None
    exp_5: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_44: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_44, view_114);  clone_44 = view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_115: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_11, [1, 16, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_116: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_45, [1, 128, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_117: "f32[128, 1024]" = torch.ops.aten.view.default(view_116, [128, 1024]);  view_116 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_33: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg90_1, view_117, permute_63);  arg90_1 = view_117 = permute_63 = None
    view_118: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_33, [1, 128, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_46: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_35: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_32, clone_46);  add_32 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_17: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_35, getitem_23);  getitem_23 = None
    mul_30: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_31: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_30, arg91_1);  mul_30 = arg91_1 = None
    add_37: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_31, arg92_1);  mul_31 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_119: "f32[128, 1024]" = torch.ops.aten.view.default(add_37, [128, 1024]);  add_37 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_34: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg94_1, view_119, permute_64);  arg94_1 = view_119 = permute_64 = None
    view_120: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_34, [1, 128, 4096]);  addmm_34 = None
    relu_5: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_47: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_5);  relu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_121: "f32[128, 4096]" = torch.ops.aten.view.default(clone_47, [128, 4096]);  clone_47 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    addmm_35: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg96_1, view_121, permute_65);  arg96_1 = view_121 = permute_65 = None
    view_122: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 128, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_48: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_122);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_35, clone_48);  add_35 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_18: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_38, getitem_25);  getitem_25 = None
    mul_32: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_33: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, arg97_1);  mul_32 = arg97_1 = None
    add_40: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_33, arg98_1);  mul_33 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_123: "f32[128, 1024]" = torch.ops.aten.view.default(add_40, [128, 1024])
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    addmm_36: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg100_1, view_123, permute_66);  arg100_1 = view_123 = permute_66 = None
    view_124: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_36, [1, 128, 1024]);  addmm_36 = None
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_124, 0.125);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_125: "f32[128, 1024]" = torch.ops.aten.view.default(add_40, [128, 1024])
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    addmm_37: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg102_1, view_125, permute_67);  arg102_1 = view_125 = permute_67 = None
    view_126: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_37, [1, 128, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_126, [1, -1, 16, 64]);  view_126 = None
    permute_68: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
    clone_49: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_128: "f32[128, 1024]" = torch.ops.aten.view.default(add_40, [128, 1024]);  add_40 = None
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    addmm_38: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_128, permute_69);  arg104_1 = view_128 = permute_69 = None
    view_129: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_38, [1, 128, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_129, [1, -1, 16, 64]);  view_129 = None
    permute_70: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_50: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_34, [1, 128, 16, 64]);  mul_34 = None
    permute_71: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    clone_51: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_132: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_51, [16, -1, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_133: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_49, [16, -1, 64]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_134: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_50, [16, -1, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_132, permute_72);  view_132 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_12, [-1], True)
    sub_19: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_12, amax_6);  bmm_12 = amax_6 = None
    exp_6: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_52: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_52, view_134);  clone_52 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_135: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_13, [1, 16, 128, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_136: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_53, [1, 128, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_137: "f32[128, 1024]" = torch.ops.aten.view.default(view_136, [128, 1024]);  view_136 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_39: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg106_1, view_137, permute_74);  arg106_1 = view_137 = permute_74 = None
    view_138: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_39, [1, 128, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_54: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_41: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_38, clone_54);  add_38 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_42: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_20: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_27);  getitem_27 = None
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_35, arg107_1);  mul_35 = arg107_1 = None
    add_43: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_36, arg108_1);  mul_36 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_139: "f32[128, 1024]" = torch.ops.aten.view.default(add_43, [128, 1024]);  add_43 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_40: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg110_1, view_139, permute_75);  arg110_1 = view_139 = permute_75 = None
    view_140: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_40, [1, 128, 4096]);  addmm_40 = None
    relu_6: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_55: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_6);  relu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_141: "f32[128, 4096]" = torch.ops.aten.view.default(clone_55, [128, 4096]);  clone_55 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    addmm_41: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg112_1, view_141, permute_76);  arg112_1 = view_141 = permute_76 = None
    view_142: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 128, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_56: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_142);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_44: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_41, clone_56);  add_41 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_45: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_21: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_44, getitem_29);  getitem_29 = None
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_38: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_37, arg113_1);  mul_37 = arg113_1 = None
    add_46: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_38, arg114_1);  mul_38 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_143: "f32[128, 1024]" = torch.ops.aten.view.default(add_46, [128, 1024])
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_42: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg116_1, view_143, permute_77);  arg116_1 = view_143 = permute_77 = None
    view_144: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_42, [1, 128, 1024]);  addmm_42 = None
    mul_39: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_144, 0.125);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_145: "f32[128, 1024]" = torch.ops.aten.view.default(add_46, [128, 1024])
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_43: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg118_1, view_145, permute_78);  arg118_1 = view_145 = permute_78 = None
    view_146: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_43, [1, 128, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_147: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_146, [1, -1, 16, 64]);  view_146 = None
    permute_79: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    clone_57: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_148: "f32[128, 1024]" = torch.ops.aten.view.default(add_46, [128, 1024]);  add_46 = None
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_44: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg120_1, view_148, permute_80);  arg120_1 = view_148 = permute_80 = None
    view_149: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_44, [1, 128, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_150: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_149, [1, -1, 16, 64]);  view_149 = None
    permute_81: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    clone_58: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_151: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_39, [1, 128, 16, 64]);  mul_39 = None
    permute_82: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    clone_59: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_152: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_59, [16, -1, 64]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_153: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_57, [16, -1, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_154: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_58, [16, -1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_83: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_152, permute_83);  view_152 = permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_14, [-1], True)
    sub_22: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_14, amax_7);  bmm_14 = amax_7 = None
    exp_7: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_60: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_60, view_154);  clone_60 = view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_155: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_15, [1, 16, 128, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_156: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_61, [1, 128, 1024]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_157: "f32[128, 1024]" = torch.ops.aten.view.default(view_156, [128, 1024]);  view_156 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_45: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg122_1, view_157, permute_85);  arg122_1 = view_157 = permute_85 = None
    view_158: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_45, [1, 128, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_62: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_47: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_44, clone_62);  add_44 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_48: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_23: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_47, getitem_31);  getitem_31 = None
    mul_40: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_41: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_40, arg123_1);  mul_40 = arg123_1 = None
    add_49: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_41, arg124_1);  mul_41 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_159: "f32[128, 1024]" = torch.ops.aten.view.default(add_49, [128, 1024]);  add_49 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_46: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg126_1, view_159, permute_86);  arg126_1 = view_159 = permute_86 = None
    view_160: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_46, [1, 128, 4096]);  addmm_46 = None
    relu_7: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_160);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_63: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_7);  relu_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_161: "f32[128, 4096]" = torch.ops.aten.view.default(clone_63, [128, 4096]);  clone_63 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_47: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg128_1, view_161, permute_87);  arg128_1 = view_161 = permute_87 = None
    view_162: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 128, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_64: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_50: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_47, clone_64);  add_47 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_51: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_24: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_50, getitem_33);  getitem_33 = None
    mul_42: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_42, arg129_1);  mul_42 = arg129_1 = None
    add_52: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_43, arg130_1);  mul_43 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_163: "f32[128, 1024]" = torch.ops.aten.view.default(add_52, [128, 1024])
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_48: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg132_1, view_163, permute_88);  arg132_1 = view_163 = permute_88 = None
    view_164: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_48, [1, 128, 1024]);  addmm_48 = None
    mul_44: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_164, 0.125);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_165: "f32[128, 1024]" = torch.ops.aten.view.default(add_52, [128, 1024])
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_49: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg134_1, view_165, permute_89);  arg134_1 = view_165 = permute_89 = None
    view_166: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_49, [1, 128, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_167: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_166, [1, -1, 16, 64]);  view_166 = None
    permute_90: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    clone_65: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_168: "f32[128, 1024]" = torch.ops.aten.view.default(add_52, [128, 1024]);  add_52 = None
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_50: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg136_1, view_168, permute_91);  arg136_1 = view_168 = permute_91 = None
    view_169: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_50, [1, 128, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_169, [1, -1, 16, 64]);  view_169 = None
    permute_92: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_66: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_171: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_44, [1, 128, 16, 64]);  mul_44 = None
    permute_93: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    clone_67: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_172: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_67, [16, -1, 64]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_173: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_65, [16, -1, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_174: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_66, [16, -1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_94: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_172, permute_94);  view_172 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_16, [-1], True)
    sub_25: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_16, amax_8);  bmm_16 = amax_8 = None
    exp_8: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_68: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_68, view_174);  clone_68 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_175: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_17, [1, 16, 128, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_69: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_176: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_69, [1, 128, 1024]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_177: "f32[128, 1024]" = torch.ops.aten.view.default(view_176, [128, 1024]);  view_176 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_51: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_177, permute_96);  arg138_1 = view_177 = permute_96 = None
    view_178: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_51, [1, 128, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_70: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_53: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_50, clone_70);  add_50 = clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    add_54: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_26: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_35);  getitem_35 = None
    mul_45: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_46: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_45, arg139_1);  mul_45 = arg139_1 = None
    add_55: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_46, arg140_1);  mul_46 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_179: "f32[128, 1024]" = torch.ops.aten.view.default(add_55, [128, 1024]);  add_55 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_52: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg142_1, view_179, permute_97);  arg142_1 = view_179 = permute_97 = None
    view_180: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_52, [1, 128, 4096]);  addmm_52 = None
    relu_8: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_71: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_8);  relu_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_181: "f32[128, 4096]" = torch.ops.aten.view.default(clone_71, [128, 4096]);  clone_71 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_53: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg144_1, view_181, permute_98);  arg144_1 = view_181 = permute_98 = None
    view_182: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 128, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_72: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_182);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_56: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_53, clone_72);  add_53 = clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_27: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_56, getitem_37);  getitem_37 = None
    mul_47: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_48: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_47, arg145_1);  mul_47 = arg145_1 = None
    add_58: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_48, arg146_1);  mul_48 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_183: "f32[128, 1024]" = torch.ops.aten.view.default(add_58, [128, 1024])
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_54: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg148_1, view_183, permute_99);  arg148_1 = view_183 = permute_99 = None
    view_184: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_54, [1, 128, 1024]);  addmm_54 = None
    mul_49: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_184, 0.125);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_185: "f32[128, 1024]" = torch.ops.aten.view.default(add_58, [128, 1024])
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_55: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg150_1, view_185, permute_100);  arg150_1 = view_185 = permute_100 = None
    view_186: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_55, [1, 128, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_186, [1, -1, 16, 64]);  view_186 = None
    permute_101: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    clone_73: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_188: "f32[128, 1024]" = torch.ops.aten.view.default(add_58, [128, 1024]);  add_58 = None
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_56: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg152_1, view_188, permute_102);  arg152_1 = view_188 = permute_102 = None
    view_189: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_56, [1, 128, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_190: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_189, [1, -1, 16, 64]);  view_189 = None
    permute_103: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_74: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_191: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_49, [1, 128, 16, 64]);  mul_49 = None
    permute_104: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    clone_75: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_192: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_75, [16, -1, 64]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_193: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_73, [16, -1, 64]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_194: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_74, [16, -1, 64]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_105: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_192, permute_105);  view_192 = permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_18, [-1], True)
    sub_28: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_18, amax_9);  bmm_18 = amax_9 = None
    exp_9: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_76: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_76, view_194);  clone_76 = view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_195: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_19, [1, 16, 128, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_77: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_196: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_77, [1, 128, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_197: "f32[128, 1024]" = torch.ops.aten.view.default(view_196, [128, 1024]);  view_196 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_57: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg154_1, view_197, permute_107);  arg154_1 = view_197 = permute_107 = None
    view_198: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_57, [1, 128, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_78: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_198);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_59: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_56, clone_78);  add_56 = clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_29: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_59, getitem_39);  getitem_39 = None
    mul_50: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_51: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_50, arg155_1);  mul_50 = arg155_1 = None
    add_61: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_51, arg156_1);  mul_51 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_199: "f32[128, 1024]" = torch.ops.aten.view.default(add_61, [128, 1024]);  add_61 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_58: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg158_1, view_199, permute_108);  arg158_1 = view_199 = permute_108 = None
    view_200: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_58, [1, 128, 4096]);  addmm_58 = None
    relu_9: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_200);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_79: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_9);  relu_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_201: "f32[128, 4096]" = torch.ops.aten.view.default(clone_79, [128, 4096]);  clone_79 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_59: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg160_1, view_201, permute_109);  arg160_1 = view_201 = permute_109 = None
    view_202: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 128, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_80: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_202);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_62: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_59, clone_80);  add_59 = clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    add_63: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_30: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_62, getitem_41);  getitem_41 = None
    mul_52: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_53: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg161_1);  mul_52 = arg161_1 = None
    add_64: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg162_1);  mul_53 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_203: "f32[128, 1024]" = torch.ops.aten.view.default(add_64, [128, 1024])
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_60: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg164_1, view_203, permute_110);  arg164_1 = view_203 = permute_110 = None
    view_204: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_60, [1, 128, 1024]);  addmm_60 = None
    mul_54: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_204, 0.125);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_205: "f32[128, 1024]" = torch.ops.aten.view.default(add_64, [128, 1024])
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_61: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg166_1, view_205, permute_111);  arg166_1 = view_205 = permute_111 = None
    view_206: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_61, [1, 128, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_206, [1, -1, 16, 64]);  view_206 = None
    permute_112: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_81: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_208: "f32[128, 1024]" = torch.ops.aten.view.default(add_64, [128, 1024]);  add_64 = None
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_62: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_208, permute_113);  arg168_1 = view_208 = permute_113 = None
    view_209: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_62, [1, 128, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_210: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_209, [1, -1, 16, 64]);  view_209 = None
    permute_114: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_82: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_211: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_54, [1, 128, 16, 64]);  mul_54 = None
    permute_115: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
    clone_83: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_212: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_83, [16, -1, 64]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_213: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_81, [16, -1, 64]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_214: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_82, [16, -1, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_116: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_212, permute_116);  view_212 = permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_20, [-1], True)
    sub_31: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_20, amax_10);  bmm_20 = amax_10 = None
    exp_10: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_84: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_84, view_214);  clone_84 = view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_215: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_21, [1, 16, 128, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_85: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_216: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_85, [1, 128, 1024]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_217: "f32[128, 1024]" = torch.ops.aten.view.default(view_216, [128, 1024]);  view_216 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_63: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_217, permute_118);  arg170_1 = view_217 = permute_118 = None
    view_218: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_63, [1, 128, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_86: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_218);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_65: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_62, clone_86);  add_62 = clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    add_66: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_32: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_43);  getitem_43 = None
    mul_55: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_56: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_55, arg171_1);  mul_55 = arg171_1 = None
    add_67: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_56, arg172_1);  mul_56 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_219: "f32[128, 1024]" = torch.ops.aten.view.default(add_67, [128, 1024]);  add_67 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_64: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg174_1, view_219, permute_119);  arg174_1 = view_219 = permute_119 = None
    view_220: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_64, [1, 128, 4096]);  addmm_64 = None
    relu_10: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_220);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_87: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_10);  relu_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_221: "f32[128, 4096]" = torch.ops.aten.view.default(clone_87, [128, 4096]);  clone_87 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_65: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg176_1, view_221, permute_120);  arg176_1 = view_221 = permute_120 = None
    view_222: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 128, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_88: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_222);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_68: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_65, clone_88);  add_65 = clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    add_69: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_33: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_68, getitem_45);  getitem_45 = None
    mul_57: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_58: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg177_1);  mul_57 = arg177_1 = None
    add_70: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg178_1);  mul_58 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_223: "f32[128, 1024]" = torch.ops.aten.view.default(add_70, [128, 1024])
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_66: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg180_1, view_223, permute_121);  arg180_1 = view_223 = permute_121 = None
    view_224: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_66, [1, 128, 1024]);  addmm_66 = None
    mul_59: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_224, 0.125);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_225: "f32[128, 1024]" = torch.ops.aten.view.default(add_70, [128, 1024])
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_67: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg182_1, view_225, permute_122);  arg182_1 = view_225 = permute_122 = None
    view_226: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_67, [1, 128, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_227: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_226, [1, -1, 16, 64]);  view_226 = None
    permute_123: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    clone_89: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_228: "f32[128, 1024]" = torch.ops.aten.view.default(add_70, [128, 1024]);  add_70 = None
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_68: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg184_1, view_228, permute_124);  arg184_1 = view_228 = permute_124 = None
    view_229: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_68, [1, 128, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_230: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_229, [1, -1, 16, 64]);  view_229 = None
    permute_125: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    clone_90: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_231: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_59, [1, 128, 16, 64]);  mul_59 = None
    permute_126: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
    clone_91: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_232: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_91, [16, -1, 64]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_233: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_89, [16, -1, 64]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_234: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_90, [16, -1, 64]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_127: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_232, permute_127);  view_232 = permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_22, [-1], True)
    sub_34: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_22, amax_11);  bmm_22 = amax_11 = None
    exp_11: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_92: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_92, view_234);  clone_92 = view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_235: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_23, [1, 16, 128, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_93: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_236: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_93, [1, 128, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_237: "f32[128, 1024]" = torch.ops.aten.view.default(view_236, [128, 1024]);  view_236 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_69: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg186_1, view_237, permute_129);  arg186_1 = view_237 = permute_129 = None
    view_238: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_69, [1, 128, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_94: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_238);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_71: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_68, clone_94);  add_68 = clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    add_72: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_35: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_71, getitem_47);  getitem_47 = None
    mul_60: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_61: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, arg187_1);  mul_60 = arg187_1 = None
    add_73: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_61, arg188_1);  mul_61 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_239: "f32[128, 1024]" = torch.ops.aten.view.default(add_73, [128, 1024]);  add_73 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_70: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg190_1, view_239, permute_130);  arg190_1 = view_239 = permute_130 = None
    view_240: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_70, [1, 128, 4096]);  addmm_70 = None
    relu_11: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_240);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_95: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_11);  relu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_241: "f32[128, 4096]" = torch.ops.aten.view.default(clone_95, [128, 4096]);  clone_95 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    addmm_71: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg192_1, view_241, permute_131);  arg192_1 = view_241 = permute_131 = None
    view_242: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 128, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_96: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_242);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_74: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_71, clone_96);  add_71 = clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:852, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    add_75: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_36: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_74, getitem_49);  add_74 = getitem_49 = None
    mul_62: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_63: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_62, arg193_1);  mul_62 = arg193_1 = None
    add_76: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_63, arg194_1);  mul_63 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:990, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_243: "i64[1, 128]" = torch.ops.aten.view.default(arg514_1, [-1, 128]);  arg514_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1000, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding_1: "f32[1, 128, 1024]" = torch.ops.aten.embedding.default(arg195_1, view_243, 1);  arg195_1 = None
    mul_64: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_1, 32.0);  embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:82, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:83, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:84, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_77: "i64[128]" = torch.ops.aten.add.Tensor(iota, 1)
    view_244: "i64[128, 1]" = torch.ops.aten.view.default(add_77, [128, 1]);  add_77 = None
    lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota, view_244);  iota = view_244 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[128, 128]" = torch.ops.aten.where.self(lt, scalar_tensor, full);  lt = scalar_tensor = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:113, code: mask = input_ids.ne(padding_idx).int()
    ne_1: "b8[1, 128]" = torch.ops.aten.ne.Scalar(view_243, 1);  view_243 = None
    convert_element_type_3: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:114, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum_1: "i64[1, 128]" = torch.ops.aten.cumsum.default(convert_element_type_3, 1)
    convert_element_type_4: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(cumsum_1, torch.int32);  cumsum_1 = None
    add_78: "i32[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_4, 0);  convert_element_type_4 = None
    mul_65: "i32[1, 128]" = torch.ops.aten.mul.Tensor(add_78, convert_element_type_3);  add_78 = convert_element_type_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:115, code: return incremental_indices.long() + padding_idx
    convert_element_type_5: "i64[1, 128]" = torch.ops.prims.convert_element_type.default(mul_65, torch.int64);  mul_65 = None
    add_79: "i64[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1);  convert_element_type_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:176, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_245: "i64[128]" = torch.ops.aten.view.default(add_79, [-1]);  add_79 = None
    index_1: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg512_1, [view_245]);  arg512_1 = view_245 = None
    view_246: "f32[1, 128, 1024]" = torch.ops.aten.view.default(index_1, [1, 128, 1024]);  index_1 = None
    alias_2: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(view_246);  view_246 = None
    alias_3: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1028, code: hidden_states = inputs_embeds + positions
    add_80: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_64, alias_3);  mul_64 = alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1030, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_97: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(add_80);  add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_37: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(clone_97, getitem_51);  getitem_51 = None
    mul_66: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
    mul_67: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_66, arg196_1);  mul_66 = arg196_1 = None
    add_82: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_67, arg197_1);  mul_67 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_247: "f32[128, 1024]" = torch.ops.aten.view.default(add_82, [128, 1024])
    permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    addmm_72: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg199_1, view_247, permute_132);  arg199_1 = view_247 = permute_132 = None
    view_248: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_72, [1, 128, 1024]);  addmm_72 = None
    mul_68: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_248, 0.125);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_249: "f32[128, 1024]" = torch.ops.aten.view.default(add_82, [128, 1024])
    permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    addmm_73: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg201_1, view_249, permute_133);  arg201_1 = view_249 = permute_133 = None
    view_250: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_73, [1, 128, 1024]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_251: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_250, [1, -1, 16, 64]);  view_250 = None
    permute_134: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    clone_98: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_252: "f32[128, 1024]" = torch.ops.aten.view.default(add_82, [128, 1024]);  add_82 = None
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    addmm_74: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg203_1, view_252, permute_135);  arg203_1 = view_252 = permute_135 = None
    view_253: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_74, [1, 128, 1024]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_254: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_253, [1, -1, 16, 64]);  view_253 = None
    permute_136: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
    clone_99: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_255: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_68, [1, 128, 16, 64]);  mul_68 = None
    permute_137: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    clone_100: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_256: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_100, [16, -1, 64]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_257: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_98, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_258: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_99, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_138: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_256, permute_138);  view_256 = permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_259: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    slice_3: "f32[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, 9223372036854775807);  unsqueeze_3 = None
    slice_4: "f32[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807);  slice_3 = None
    expand_1: "f32[1, 1, 128, 128]" = torch.ops.aten.expand.default(slice_4, [1, 1, 128, 128]);  slice_4 = None
    add_83: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_259, expand_1);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_260: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_83, [16, 128, 128]);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_260, [-1], True)
    sub_38: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_260, amax_12);  view_260 = amax_12 = None
    exp_12: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_101: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_101, view_258);  clone_101 = view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_261: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_25, [1, 16, 128, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_139: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_102: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_262: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_102, [1, 128, 1024]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_263: "f32[128, 1024]" = torch.ops.aten.view.default(view_262, [128, 1024]);  view_262 = None
    permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_75: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg205_1, view_263, permute_140);  arg205_1 = view_263 = permute_140 = None
    view_264: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_75, [1, 128, 1024]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_103: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_264);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_84: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(clone_97, clone_103);  clone_97 = clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    add_85: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_39: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_84, getitem_53);  getitem_53 = None
    mul_69: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
    mul_70: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_69, arg206_1);  mul_69 = arg206_1 = None
    add_86: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_70, arg207_1);  mul_70 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_265: "f32[128, 1024]" = torch.ops.aten.view.default(add_86, [128, 1024]);  add_86 = None
    permute_141: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_76: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg209_1, view_265, permute_141);  arg209_1 = view_265 = permute_141 = None
    view_266: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_76, [1, 128, 1024]);  addmm_76 = None
    mul_71: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_266, 0.125);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_267: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_142: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_77: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg211_1, view_267, permute_142);  arg211_1 = view_267 = permute_142 = None
    view_268: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_77, [1, 128, 1024]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_269: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_268, [1, -1, 16, 64]);  view_268 = None
    permute_143: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_104: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_270: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    addmm_78: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg213_1, view_270, permute_144);  arg213_1 = view_270 = permute_144 = None
    view_271: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_78, [1, 128, 1024]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_272: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_271, [1, -1, 16, 64]);  view_271 = None
    permute_145: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    clone_105: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_273: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_71, [1, 128, 16, 64]);  mul_71 = None
    permute_146: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    clone_106: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_274: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_106, [16, -1, 64]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_275: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_104, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_276: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_105, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_147: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_274, permute_147);  view_274 = permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_26, [-1], True)
    sub_40: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_26, amax_13);  bmm_26 = amax_13 = None
    exp_13: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_107: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_27: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_107, view_276);  clone_107 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_277: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_27, [1, 16, 128, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_148: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_108: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_278: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_108, [1, 128, 1024]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_279: "f32[128, 1024]" = torch.ops.aten.view.default(view_278, [128, 1024]);  view_278 = None
    permute_149: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_79: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg215_1, view_279, permute_149);  arg215_1 = view_279 = permute_149 = None
    view_280: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_79, [1, 128, 1024]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_109: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_280);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_87: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_84, clone_109);  add_84 = clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    add_88: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_41: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_87, getitem_55);  getitem_55 = None
    mul_72: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
    mul_73: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_72, arg216_1);  mul_72 = arg216_1 = None
    add_89: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_73, arg217_1);  mul_73 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_281: "f32[128, 1024]" = torch.ops.aten.view.default(add_89, [128, 1024]);  add_89 = None
    permute_150: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    addmm_80: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg219_1, view_281, permute_150);  arg219_1 = view_281 = permute_150 = None
    view_282: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_80, [1, 128, 4096]);  addmm_80 = None
    relu_12: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_282);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_110: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_12);  relu_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_283: "f32[128, 4096]" = torch.ops.aten.view.default(clone_110, [128, 4096]);  clone_110 = None
    permute_151: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    addmm_81: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg221_1, view_283, permute_151);  arg221_1 = view_283 = permute_151 = None
    view_284: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_81, [1, 128, 1024]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_111: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_284);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_90: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_87, clone_111);  add_87 = clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_90, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    add_91: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_42: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_90, getitem_57);  getitem_57 = None
    mul_74: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
    mul_75: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_74, arg222_1);  mul_74 = arg222_1 = None
    add_92: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_75, arg223_1);  mul_75 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_285: "f32[128, 1024]" = torch.ops.aten.view.default(add_92, [128, 1024])
    permute_152: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    addmm_82: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg225_1, view_285, permute_152);  arg225_1 = view_285 = permute_152 = None
    view_286: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_82, [1, 128, 1024]);  addmm_82 = None
    mul_76: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_286, 0.125);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_287: "f32[128, 1024]" = torch.ops.aten.view.default(add_92, [128, 1024])
    permute_153: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    addmm_83: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg227_1, view_287, permute_153);  arg227_1 = view_287 = permute_153 = None
    view_288: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_83, [1, 128, 1024]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_289: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_288, [1, -1, 16, 64]);  view_288 = None
    permute_154: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    clone_112: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_290: "f32[128, 1024]" = torch.ops.aten.view.default(add_92, [128, 1024]);  add_92 = None
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    addmm_84: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg229_1, view_290, permute_155);  arg229_1 = view_290 = permute_155 = None
    view_291: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_84, [1, 128, 1024]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_292: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_291, [1, -1, 16, 64]);  view_291 = None
    permute_156: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    clone_113: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_293: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_76, [1, 128, 16, 64]);  mul_76 = None
    permute_157: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    clone_114: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_294: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_114, [16, -1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_295: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_112, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_296: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_113, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_158: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_294, permute_158);  view_294 = permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_297: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    add_93: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_297, expand_1);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_298: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_93, [16, 128, 128]);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_298, [-1], True)
    sub_43: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_298, amax_14);  view_298 = amax_14 = None
    exp_14: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_115: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_115, view_296);  clone_115 = view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_299: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_29, [1, 16, 128, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_159: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_116: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    view_300: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_116, [1, 128, 1024]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_301: "f32[128, 1024]" = torch.ops.aten.view.default(view_300, [128, 1024]);  view_300 = None
    permute_160: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_85: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg231_1, view_301, permute_160);  arg231_1 = view_301 = permute_160 = None
    view_302: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_85, [1, 128, 1024]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_117: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_302);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_94: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_90, clone_117);  add_90 = clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    add_95: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_44: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_94, getitem_59);  getitem_59 = None
    mul_77: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_78: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_77, arg232_1);  mul_77 = arg232_1 = None
    add_96: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_78, arg233_1);  mul_78 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_303: "f32[128, 1024]" = torch.ops.aten.view.default(add_96, [128, 1024]);  add_96 = None
    permute_161: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_86: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg235_1, view_303, permute_161);  arg235_1 = view_303 = permute_161 = None
    view_304: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_86, [1, 128, 1024]);  addmm_86 = None
    mul_79: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_304, 0.125);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_305: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    addmm_87: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg237_1, view_305, permute_162);  arg237_1 = view_305 = permute_162 = None
    view_306: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_87, [1, 128, 1024]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_307: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_306, [1, -1, 16, 64]);  view_306 = None
    permute_163: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    clone_118: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_308: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_164: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    addmm_88: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg239_1, view_308, permute_164);  arg239_1 = view_308 = permute_164 = None
    view_309: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_88, [1, 128, 1024]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_310: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_309, [1, -1, 16, 64]);  view_309 = None
    permute_165: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    clone_119: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_311: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_79, [1, 128, 16, 64]);  mul_79 = None
    permute_166: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
    clone_120: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_312: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_120, [16, -1, 64]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_313: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_118, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_314: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_119, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_167: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_312, permute_167);  view_312 = permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_30, [-1], True)
    sub_45: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_30, amax_15);  bmm_30 = amax_15 = None
    exp_15: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_16: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_121: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_31: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_121, view_314);  clone_121 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_315: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_31, [1, 16, 128, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_168: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_122: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_316: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_122, [1, 128, 1024]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_317: "f32[128, 1024]" = torch.ops.aten.view.default(view_316, [128, 1024]);  view_316 = None
    permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_89: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg241_1, view_317, permute_169);  arg241_1 = view_317 = permute_169 = None
    view_318: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_89, [1, 128, 1024]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_123: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_318);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_97: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_94, clone_123);  add_94 = clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    add_98: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_46: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_61);  getitem_61 = None
    mul_80: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
    mul_81: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_80, arg242_1);  mul_80 = arg242_1 = None
    add_99: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_81, arg243_1);  mul_81 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_319: "f32[128, 1024]" = torch.ops.aten.view.default(add_99, [128, 1024]);  add_99 = None
    permute_170: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    addmm_90: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg245_1, view_319, permute_170);  arg245_1 = view_319 = permute_170 = None
    view_320: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_90, [1, 128, 4096]);  addmm_90 = None
    relu_13: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_320);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_124: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_13);  relu_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_321: "f32[128, 4096]" = torch.ops.aten.view.default(clone_124, [128, 4096]);  clone_124 = None
    permute_171: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    addmm_91: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg247_1, view_321, permute_171);  arg247_1 = view_321 = permute_171 = None
    view_322: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_91, [1, 128, 1024]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_125: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_322);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_100: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_97, clone_125);  add_97 = clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    add_101: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_47: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_100, getitem_63);  getitem_63 = None
    mul_82: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_83: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_82, arg248_1);  mul_82 = arg248_1 = None
    add_102: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_83, arg249_1);  mul_83 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_323: "f32[128, 1024]" = torch.ops.aten.view.default(add_102, [128, 1024])
    permute_172: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_92: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg251_1, view_323, permute_172);  arg251_1 = view_323 = permute_172 = None
    view_324: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_92, [1, 128, 1024]);  addmm_92 = None
    mul_84: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_324, 0.125);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_325: "f32[128, 1024]" = torch.ops.aten.view.default(add_102, [128, 1024])
    permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
    addmm_93: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg253_1, view_325, permute_173);  arg253_1 = view_325 = permute_173 = None
    view_326: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_93, [1, 128, 1024]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_327: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_326, [1, -1, 16, 64]);  view_326 = None
    permute_174: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    clone_126: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_328: "f32[128, 1024]" = torch.ops.aten.view.default(add_102, [128, 1024]);  add_102 = None
    permute_175: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_94: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg255_1, view_328, permute_175);  arg255_1 = view_328 = permute_175 = None
    view_329: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_94, [1, 128, 1024]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_330: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_329, [1, -1, 16, 64]);  view_329 = None
    permute_176: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    clone_127: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_331: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_84, [1, 128, 16, 64]);  mul_84 = None
    permute_177: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    clone_128: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_332: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_128, [16, -1, 64]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_333: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_126, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_334: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_127, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_178: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_332, permute_178);  view_332 = permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_335: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    add_103: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_335, expand_1);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_336: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_103, [16, 128, 128]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_336, [-1], True)
    sub_48: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_336, amax_16);  view_336 = amax_16 = None
    exp_16: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_17: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_129: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_129, view_334);  clone_129 = view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_337: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_33, [1, 16, 128, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_179: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_130: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_338: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_130, [1, 128, 1024]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_339: "f32[128, 1024]" = torch.ops.aten.view.default(view_338, [128, 1024]);  view_338 = None
    permute_180: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    addmm_95: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg257_1, view_339, permute_180);  arg257_1 = view_339 = permute_180 = None
    view_340: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_95, [1, 128, 1024]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_131: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_340);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_104: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_100, clone_131);  add_100 = clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_49: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_104, getitem_65);  getitem_65 = None
    mul_85: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
    mul_86: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_85, arg258_1);  mul_85 = arg258_1 = None
    add_106: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_86, arg259_1);  mul_86 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_341: "f32[128, 1024]" = torch.ops.aten.view.default(add_106, [128, 1024]);  add_106 = None
    permute_181: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm_96: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg261_1, view_341, permute_181);  arg261_1 = view_341 = permute_181 = None
    view_342: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_96, [1, 128, 1024]);  addmm_96 = None
    mul_87: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_342, 0.125);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_343: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_182: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
    addmm_97: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg263_1, view_343, permute_182);  arg263_1 = view_343 = permute_182 = None
    view_344: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_97, [1, 128, 1024]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_345: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_344, [1, -1, 16, 64]);  view_344 = None
    permute_183: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_132: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_346: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    addmm_98: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg265_1, view_346, permute_184);  arg265_1 = view_346 = permute_184 = None
    view_347: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_98, [1, 128, 1024]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_348: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_347, [1, -1, 16, 64]);  view_347 = None
    permute_185: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
    clone_133: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_349: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_87, [1, 128, 16, 64]);  mul_87 = None
    permute_186: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    clone_134: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_350: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_134, [16, -1, 64]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_351: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_132, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_352: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_133, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_187: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_350, permute_187);  view_350 = permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_34, [-1], True)
    sub_50: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_34, amax_17);  bmm_34 = amax_17 = None
    exp_17: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_18: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_135: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_35: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_135, view_352);  clone_135 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_353: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_35, [1, 16, 128, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_188: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_136: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_354: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_136, [1, 128, 1024]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_355: "f32[128, 1024]" = torch.ops.aten.view.default(view_354, [128, 1024]);  view_354 = None
    permute_189: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    addmm_99: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg267_1, view_355, permute_189);  arg267_1 = view_355 = permute_189 = None
    view_356: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_99, [1, 128, 1024]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_137: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_356);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_107: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_104, clone_137);  add_104 = clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_51: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_107, getitem_67);  getitem_67 = None
    mul_88: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = rsqrt_33 = None
    mul_89: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_88, arg268_1);  mul_88 = arg268_1 = None
    add_109: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_89, arg269_1);  mul_89 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_357: "f32[128, 1024]" = torch.ops.aten.view.default(add_109, [128, 1024]);  add_109 = None
    permute_190: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_100: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg271_1, view_357, permute_190);  arg271_1 = view_357 = permute_190 = None
    view_358: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_100, [1, 128, 4096]);  addmm_100 = None
    relu_14: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_358);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_138: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_14);  relu_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_359: "f32[128, 4096]" = torch.ops.aten.view.default(clone_138, [128, 4096]);  clone_138 = None
    permute_191: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
    addmm_101: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg273_1, view_359, permute_191);  arg273_1 = view_359 = permute_191 = None
    view_360: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_101, [1, 128, 1024]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_139: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_360);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_110: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_107, clone_139);  add_107 = clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_52: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_110, getitem_69);  getitem_69 = None
    mul_90: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
    mul_91: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_90, arg274_1);  mul_90 = arg274_1 = None
    add_112: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_91, arg275_1);  mul_91 = arg275_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_361: "f32[128, 1024]" = torch.ops.aten.view.default(add_112, [128, 1024])
    permute_192: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    addmm_102: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg277_1, view_361, permute_192);  arg277_1 = view_361 = permute_192 = None
    view_362: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_102, [1, 128, 1024]);  addmm_102 = None
    mul_92: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_362, 0.125);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_363: "f32[128, 1024]" = torch.ops.aten.view.default(add_112, [128, 1024])
    permute_193: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    addmm_103: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg279_1, view_363, permute_193);  arg279_1 = view_363 = permute_193 = None
    view_364: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_103, [1, 128, 1024]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_365: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_364, [1, -1, 16, 64]);  view_364 = None
    permute_194: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    clone_140: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_366: "f32[128, 1024]" = torch.ops.aten.view.default(add_112, [128, 1024]);  add_112 = None
    permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
    addmm_104: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg281_1, view_366, permute_195);  arg281_1 = view_366 = permute_195 = None
    view_367: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_104, [1, 128, 1024]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_368: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_367, [1, -1, 16, 64]);  view_367 = None
    permute_196: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_141: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_369: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_92, [1, 128, 16, 64]);  mul_92 = None
    permute_197: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
    clone_142: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_370: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_142, [16, -1, 64]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_371: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_140, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_372: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_141, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_198: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_370, permute_198);  view_370 = permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_373: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    add_113: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_373, expand_1);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_374: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_113, [16, 128, 128]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_374, [-1], True)
    sub_53: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_374, amax_18);  view_374 = amax_18 = None
    exp_18: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
    sum_19: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_143: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_37: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_143, view_372);  clone_143 = view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_375: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_37, [1, 16, 128, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_199: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_144: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_376: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_144, [1, 128, 1024]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_377: "f32[128, 1024]" = torch.ops.aten.view.default(view_376, [128, 1024]);  view_376 = None
    permute_200: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
    addmm_105: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg283_1, view_377, permute_200);  arg283_1 = view_377 = permute_200 = None
    view_378: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_105, [1, 128, 1024]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_145: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_378);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_114: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_110, clone_145);  add_110 = clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    add_115: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_54: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_114, getitem_71);  getitem_71 = None
    mul_93: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = rsqrt_35 = None
    mul_94: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_93, arg284_1);  mul_93 = arg284_1 = None
    add_116: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_94, arg285_1);  mul_94 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_379: "f32[128, 1024]" = torch.ops.aten.view.default(add_116, [128, 1024]);  add_116 = None
    permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
    addmm_106: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg287_1, view_379, permute_201);  arg287_1 = view_379 = permute_201 = None
    view_380: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_106, [1, 128, 1024]);  addmm_106 = None
    mul_95: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_380, 0.125);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_381: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_202: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
    addmm_107: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg289_1, view_381, permute_202);  arg289_1 = view_381 = permute_202 = None
    view_382: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_107, [1, 128, 1024]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_383: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_382, [1, -1, 16, 64]);  view_382 = None
    permute_203: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    clone_146: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_384: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_204: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
    addmm_108: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg291_1, view_384, permute_204);  arg291_1 = view_384 = permute_204 = None
    view_385: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_108, [1, 128, 1024]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_386: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_385, [1, -1, 16, 64]);  view_385 = None
    permute_205: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
    clone_147: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_387: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_95, [1, 128, 16, 64]);  mul_95 = None
    permute_206: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    clone_148: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_388: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_148, [16, -1, 64]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_389: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_146, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_390: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_147, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_207: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_388, permute_207);  view_388 = permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_38, [-1], True)
    sub_55: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_38, amax_19);  bmm_38 = amax_19 = None
    exp_19: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_20: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_149: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_39: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_149, view_390);  clone_149 = view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_391: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_39, [1, 16, 128, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_208: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_150: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    view_392: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_150, [1, 128, 1024]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_393: "f32[128, 1024]" = torch.ops.aten.view.default(view_392, [128, 1024]);  view_392 = None
    permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
    addmm_109: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg293_1, view_393, permute_209);  arg293_1 = view_393 = permute_209 = None
    view_394: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_109, [1, 128, 1024]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_151: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_394);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_117: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_114, clone_151);  add_114 = clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    add_118: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_56: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_73);  getitem_73 = None
    mul_96: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_36);  sub_56 = rsqrt_36 = None
    mul_97: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_96, arg294_1);  mul_96 = arg294_1 = None
    add_119: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_97, arg295_1);  mul_97 = arg295_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_395: "f32[128, 1024]" = torch.ops.aten.view.default(add_119, [128, 1024]);  add_119 = None
    permute_210: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
    addmm_110: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg297_1, view_395, permute_210);  arg297_1 = view_395 = permute_210 = None
    view_396: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_110, [1, 128, 4096]);  addmm_110 = None
    relu_15: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_396);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_152: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_15);  relu_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_397: "f32[128, 4096]" = torch.ops.aten.view.default(clone_152, [128, 4096]);  clone_152 = None
    permute_211: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
    addmm_111: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg299_1, view_397, permute_211);  arg299_1 = view_397 = permute_211 = None
    view_398: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_111, [1, 128, 1024]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_153: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_398);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_120: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_117, clone_153);  add_117 = clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    add_121: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_57: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_120, getitem_75);  getitem_75 = None
    mul_98: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = rsqrt_37 = None
    mul_99: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_98, arg300_1);  mul_98 = arg300_1 = None
    add_122: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_99, arg301_1);  mul_99 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_399: "f32[128, 1024]" = torch.ops.aten.view.default(add_122, [128, 1024])
    permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
    addmm_112: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg303_1, view_399, permute_212);  arg303_1 = view_399 = permute_212 = None
    view_400: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_112, [1, 128, 1024]);  addmm_112 = None
    mul_100: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_400, 0.125);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_401: "f32[128, 1024]" = torch.ops.aten.view.default(add_122, [128, 1024])
    permute_213: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
    addmm_113: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg305_1, view_401, permute_213);  arg305_1 = view_401 = permute_213 = None
    view_402: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_113, [1, 128, 1024]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_403: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_402, [1, -1, 16, 64]);  view_402 = None
    permute_214: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    clone_154: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_404: "f32[128, 1024]" = torch.ops.aten.view.default(add_122, [128, 1024]);  add_122 = None
    permute_215: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
    addmm_114: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg307_1, view_404, permute_215);  arg307_1 = view_404 = permute_215 = None
    view_405: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_114, [1, 128, 1024]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_406: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_405, [1, -1, 16, 64]);  view_405 = None
    permute_216: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    clone_155: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_407: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_100, [1, 128, 16, 64]);  mul_100 = None
    permute_217: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
    clone_156: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_408: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_156, [16, -1, 64]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_409: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_154, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_410: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_155, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_218: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_408, permute_218);  view_408 = permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_411: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    add_123: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_411, expand_1);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_412: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_123, [16, 128, 128]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_412, [-1], True)
    sub_58: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_412, amax_20);  view_412 = amax_20 = None
    exp_20: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_21: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_157: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_41: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_157, view_410);  clone_157 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_413: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_41, [1, 16, 128, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_219: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_158: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_414: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_158, [1, 128, 1024]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_415: "f32[128, 1024]" = torch.ops.aten.view.default(view_414, [128, 1024]);  view_414 = None
    permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
    addmm_115: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg309_1, view_415, permute_220);  arg309_1 = view_415 = permute_220 = None
    view_416: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_115, [1, 128, 1024]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_159: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_416);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_124: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_120, clone_159);  add_120 = clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_124, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    add_125: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    sub_59: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_124, getitem_77);  getitem_77 = None
    mul_101: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_38);  sub_59 = rsqrt_38 = None
    mul_102: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_101, arg310_1);  mul_101 = arg310_1 = None
    add_126: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_102, arg311_1);  mul_102 = arg311_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_417: "f32[128, 1024]" = torch.ops.aten.view.default(add_126, [128, 1024]);  add_126 = None
    permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
    addmm_116: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg313_1, view_417, permute_221);  arg313_1 = view_417 = permute_221 = None
    view_418: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_116, [1, 128, 1024]);  addmm_116 = None
    mul_103: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_418, 0.125);  view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_419: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_222: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
    addmm_117: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg315_1, view_419, permute_222);  arg315_1 = view_419 = permute_222 = None
    view_420: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_117, [1, 128, 1024]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_421: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_420, [1, -1, 16, 64]);  view_420 = None
    permute_223: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    clone_160: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_422: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_224: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
    addmm_118: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg317_1, view_422, permute_224);  arg317_1 = view_422 = permute_224 = None
    view_423: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_118, [1, 128, 1024]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_424: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_423, [1, -1, 16, 64]);  view_423 = None
    permute_225: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
    clone_161: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_425: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_103, [1, 128, 16, 64]);  mul_103 = None
    permute_226: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    clone_162: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_426: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_162, [16, -1, 64]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_427: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_160, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_428: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_161, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_227: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_426, permute_227);  view_426 = permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_42, [-1], True)
    sub_60: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_42, amax_21);  bmm_42 = amax_21 = None
    exp_21: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_22: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_163: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_43: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_163, view_428);  clone_163 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_429: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_43, [1, 16, 128, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_228: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_164: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_430: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_164, [1, 128, 1024]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_431: "f32[128, 1024]" = torch.ops.aten.view.default(view_430, [128, 1024]);  view_430 = None
    permute_229: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
    addmm_119: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg319_1, view_431, permute_229);  arg319_1 = view_431 = permute_229 = None
    view_432: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_119, [1, 128, 1024]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_165: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_432);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_127: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_124, clone_165);  add_124 = clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_127, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    add_128: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_61: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_127, getitem_79);  getitem_79 = None
    mul_104: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_39);  sub_61 = rsqrt_39 = None
    mul_105: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_104, arg320_1);  mul_104 = arg320_1 = None
    add_129: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_105, arg321_1);  mul_105 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_433: "f32[128, 1024]" = torch.ops.aten.view.default(add_129, [128, 1024]);  add_129 = None
    permute_230: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
    addmm_120: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg323_1, view_433, permute_230);  arg323_1 = view_433 = permute_230 = None
    view_434: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_120, [1, 128, 4096]);  addmm_120 = None
    relu_16: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_434);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_166: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_16);  relu_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_435: "f32[128, 4096]" = torch.ops.aten.view.default(clone_166, [128, 4096]);  clone_166 = None
    permute_231: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
    addmm_121: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg325_1, view_435, permute_231);  arg325_1 = view_435 = permute_231 = None
    view_436: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_121, [1, 128, 1024]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_167: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_436);  view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_130: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_127, clone_167);  add_127 = clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    add_131: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_62: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_130, getitem_81);  getitem_81 = None
    mul_106: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_40);  sub_62 = rsqrt_40 = None
    mul_107: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_106, arg326_1);  mul_106 = arg326_1 = None
    add_132: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_107, arg327_1);  mul_107 = arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_437: "f32[128, 1024]" = torch.ops.aten.view.default(add_132, [128, 1024])
    permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
    addmm_122: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg329_1, view_437, permute_232);  arg329_1 = view_437 = permute_232 = None
    view_438: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_122, [1, 128, 1024]);  addmm_122 = None
    mul_108: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_438, 0.125);  view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_439: "f32[128, 1024]" = torch.ops.aten.view.default(add_132, [128, 1024])
    permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
    addmm_123: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg331_1, view_439, permute_233);  arg331_1 = view_439 = permute_233 = None
    view_440: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_123, [1, 128, 1024]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_441: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_440, [1, -1, 16, 64]);  view_440 = None
    permute_234: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
    clone_168: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_442: "f32[128, 1024]" = torch.ops.aten.view.default(add_132, [128, 1024]);  add_132 = None
    permute_235: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
    addmm_124: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg333_1, view_442, permute_235);  arg333_1 = view_442 = permute_235 = None
    view_443: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_124, [1, 128, 1024]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_444: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_443, [1, -1, 16, 64]);  view_443 = None
    permute_236: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
    clone_169: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_445: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_108, [1, 128, 16, 64]);  mul_108 = None
    permute_237: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    clone_170: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_446: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_170, [16, -1, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_447: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_168, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_448: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_169, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_238: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_446, permute_238);  view_446 = permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_449: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    add_133: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_449, expand_1);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_450: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_133, [16, 128, 128]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_450, [-1], True)
    sub_63: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_450, amax_22);  view_450 = amax_22 = None
    exp_22: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_23: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_171: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_45: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_171, view_448);  clone_171 = view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_451: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_45, [1, 16, 128, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_239: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_172: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
    view_452: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_172, [1, 128, 1024]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_453: "f32[128, 1024]" = torch.ops.aten.view.default(view_452, [128, 1024]);  view_452 = None
    permute_240: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
    addmm_125: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg335_1, view_453, permute_240);  arg335_1 = view_453 = permute_240 = None
    view_454: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_125, [1, 128, 1024]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_173: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_454);  view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_134: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_130, clone_173);  add_130 = clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    add_135: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_64: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_134, getitem_83);  getitem_83 = None
    mul_109: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_41);  sub_64 = rsqrt_41 = None
    mul_110: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_109, arg336_1);  mul_109 = arg336_1 = None
    add_136: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_110, arg337_1);  mul_110 = arg337_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_455: "f32[128, 1024]" = torch.ops.aten.view.default(add_136, [128, 1024]);  add_136 = None
    permute_241: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
    addmm_126: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg339_1, view_455, permute_241);  arg339_1 = view_455 = permute_241 = None
    view_456: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_126, [1, 128, 1024]);  addmm_126 = None
    mul_111: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_456, 0.125);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_457: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
    addmm_127: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg341_1, view_457, permute_242);  arg341_1 = view_457 = permute_242 = None
    view_458: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_127, [1, 128, 1024]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_459: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_458, [1, -1, 16, 64]);  view_458 = None
    permute_243: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    clone_174: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_460: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_244: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
    addmm_128: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg343_1, view_460, permute_244);  arg343_1 = view_460 = permute_244 = None
    view_461: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_128, [1, 128, 1024]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_462: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_461, [1, -1, 16, 64]);  view_461 = None
    permute_245: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    clone_175: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_463: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_111, [1, 128, 16, 64]);  mul_111 = None
    permute_246: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None
    clone_176: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_464: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_176, [16, -1, 64]);  clone_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_465: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_174, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_466: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_175, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_247: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_464, permute_247);  view_464 = permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_46, [-1], True)
    sub_65: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_46, amax_23);  bmm_46 = amax_23 = None
    exp_23: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
    sum_24: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_177: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_47: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_177, view_466);  clone_177 = view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_467: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_47, [1, 16, 128, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_248: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_178: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    view_468: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_178, [1, 128, 1024]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_469: "f32[128, 1024]" = torch.ops.aten.view.default(view_468, [128, 1024]);  view_468 = None
    permute_249: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
    addmm_129: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg345_1, view_469, permute_249);  arg345_1 = view_469 = permute_249 = None
    view_470: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_129, [1, 128, 1024]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_179: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_470);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_137: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_134, clone_179);  add_134 = clone_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    add_138: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_66: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_137, getitem_85);  getitem_85 = None
    mul_112: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_42);  sub_66 = rsqrt_42 = None
    mul_113: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_112, arg346_1);  mul_112 = arg346_1 = None
    add_139: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_113, arg347_1);  mul_113 = arg347_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_471: "f32[128, 1024]" = torch.ops.aten.view.default(add_139, [128, 1024]);  add_139 = None
    permute_250: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
    addmm_130: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg349_1, view_471, permute_250);  arg349_1 = view_471 = permute_250 = None
    view_472: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_130, [1, 128, 4096]);  addmm_130 = None
    relu_17: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_472);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_180: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_17);  relu_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_473: "f32[128, 4096]" = torch.ops.aten.view.default(clone_180, [128, 4096]);  clone_180 = None
    permute_251: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
    addmm_131: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg351_1, view_473, permute_251);  arg351_1 = view_473 = permute_251 = None
    view_474: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_131, [1, 128, 1024]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_181: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_474);  view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_140: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_137, clone_181);  add_137 = clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    add_141: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_67: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_140, getitem_87);  getitem_87 = None
    mul_114: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_43);  sub_67 = rsqrt_43 = None
    mul_115: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_114, arg352_1);  mul_114 = arg352_1 = None
    add_142: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_115, arg353_1);  mul_115 = arg353_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_475: "f32[128, 1024]" = torch.ops.aten.view.default(add_142, [128, 1024])
    permute_252: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
    addmm_132: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg355_1, view_475, permute_252);  arg355_1 = view_475 = permute_252 = None
    view_476: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_132, [1, 128, 1024]);  addmm_132 = None
    mul_116: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_476, 0.125);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_477: "f32[128, 1024]" = torch.ops.aten.view.default(add_142, [128, 1024])
    permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
    addmm_133: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg357_1, view_477, permute_253);  arg357_1 = view_477 = permute_253 = None
    view_478: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_133, [1, 128, 1024]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_479: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_478, [1, -1, 16, 64]);  view_478 = None
    permute_254: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
    clone_182: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_480: "f32[128, 1024]" = torch.ops.aten.view.default(add_142, [128, 1024]);  add_142 = None
    permute_255: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
    addmm_134: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg359_1, view_480, permute_255);  arg359_1 = view_480 = permute_255 = None
    view_481: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_134, [1, 128, 1024]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_482: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_481, [1, -1, 16, 64]);  view_481 = None
    permute_256: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    clone_183: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_483: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_116, [1, 128, 16, 64]);  mul_116 = None
    permute_257: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
    clone_184: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_484: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_184, [16, -1, 64]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_485: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_182, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_486: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_183, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_258: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_485, [0, 2, 1]);  view_485 = None
    bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_484, permute_258);  view_484 = permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_487: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None
    add_143: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_487, expand_1);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_488: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_143, [16, 128, 128]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_24: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_488, [-1], True)
    sub_68: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_488, amax_24);  view_488 = amax_24 = None
    exp_24: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_25: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_24: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_185: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_24);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_49: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_185, view_486);  clone_185 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_489: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_49, [1, 16, 128, 64]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_259: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_186: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_490: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_186, [1, 128, 1024]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_491: "f32[128, 1024]" = torch.ops.aten.view.default(view_490, [128, 1024]);  view_490 = None
    permute_260: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
    addmm_135: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg361_1, view_491, permute_260);  arg361_1 = view_491 = permute_260 = None
    view_492: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_135, [1, 128, 1024]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_187: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_492);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_144: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_140, clone_187);  add_140 = clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_69: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_144, getitem_89);  getitem_89 = None
    mul_117: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_44);  sub_69 = rsqrt_44 = None
    mul_118: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_117, arg362_1);  mul_117 = arg362_1 = None
    add_146: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_118, arg363_1);  mul_118 = arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_493: "f32[128, 1024]" = torch.ops.aten.view.default(add_146, [128, 1024]);  add_146 = None
    permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
    addmm_136: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg365_1, view_493, permute_261);  arg365_1 = view_493 = permute_261 = None
    view_494: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_136, [1, 128, 1024]);  addmm_136 = None
    mul_119: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_494, 0.125);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_495: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_262: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg366_1, [1, 0]);  arg366_1 = None
    addmm_137: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg367_1, view_495, permute_262);  arg367_1 = view_495 = permute_262 = None
    view_496: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_137, [1, 128, 1024]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_497: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_496, [1, -1, 16, 64]);  view_496 = None
    permute_263: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_188: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_498: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
    addmm_138: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg369_1, view_498, permute_264);  arg369_1 = view_498 = permute_264 = None
    view_499: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_138, [1, 128, 1024]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_500: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_499, [1, -1, 16, 64]);  view_499 = None
    permute_265: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
    clone_189: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_501: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_119, [1, 128, 16, 64]);  mul_119 = None
    permute_266: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
    clone_190: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_502: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_190, [16, -1, 64]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_503: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_188, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_504: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_189, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_267: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_503, [0, 2, 1]);  view_503 = None
    bmm_50: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_502, permute_267);  view_502 = permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_25: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_50, [-1], True)
    sub_70: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_50, amax_25);  bmm_50 = amax_25 = None
    exp_25: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_26: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
    div_25: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_191: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_25);  div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_51: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_191, view_504);  clone_191 = view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_505: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_51, [1, 16, 128, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_268: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_192: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    view_506: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_192, [1, 128, 1024]);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_507: "f32[128, 1024]" = torch.ops.aten.view.default(view_506, [128, 1024]);  view_506 = None
    permute_269: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
    addmm_139: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg371_1, view_507, permute_269);  arg371_1 = view_507 = permute_269 = None
    view_508: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_139, [1, 128, 1024]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_193: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_508);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_147: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_144, clone_193);  add_144 = clone_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    add_148: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_71: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_147, getitem_91);  getitem_91 = None
    mul_120: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_45);  sub_71 = rsqrt_45 = None
    mul_121: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_120, arg372_1);  mul_120 = arg372_1 = None
    add_149: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_121, arg373_1);  mul_121 = arg373_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_509: "f32[128, 1024]" = torch.ops.aten.view.default(add_149, [128, 1024]);  add_149 = None
    permute_270: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
    addmm_140: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg375_1, view_509, permute_270);  arg375_1 = view_509 = permute_270 = None
    view_510: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_140, [1, 128, 4096]);  addmm_140 = None
    relu_18: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_510);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_194: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_18);  relu_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_511: "f32[128, 4096]" = torch.ops.aten.view.default(clone_194, [128, 4096]);  clone_194 = None
    permute_271: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
    addmm_141: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg377_1, view_511, permute_271);  arg377_1 = view_511 = permute_271 = None
    view_512: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_141, [1, 128, 1024]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_195: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_512);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_150: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_147, clone_195);  add_147 = clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    add_151: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_72: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_150, getitem_93);  getitem_93 = None
    mul_122: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_46);  sub_72 = rsqrt_46 = None
    mul_123: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_122, arg378_1);  mul_122 = arg378_1 = None
    add_152: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_123, arg379_1);  mul_123 = arg379_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_513: "f32[128, 1024]" = torch.ops.aten.view.default(add_152, [128, 1024])
    permute_272: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
    addmm_142: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg381_1, view_513, permute_272);  arg381_1 = view_513 = permute_272 = None
    view_514: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_142, [1, 128, 1024]);  addmm_142 = None
    mul_124: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_514, 0.125);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_515: "f32[128, 1024]" = torch.ops.aten.view.default(add_152, [128, 1024])
    permute_273: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
    addmm_143: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg383_1, view_515, permute_273);  arg383_1 = view_515 = permute_273 = None
    view_516: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_143, [1, 128, 1024]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_517: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_516, [1, -1, 16, 64]);  view_516 = None
    permute_274: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    clone_196: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_518: "f32[128, 1024]" = torch.ops.aten.view.default(add_152, [128, 1024]);  add_152 = None
    permute_275: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg384_1, [1, 0]);  arg384_1 = None
    addmm_144: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg385_1, view_518, permute_275);  arg385_1 = view_518 = permute_275 = None
    view_519: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_144, [1, 128, 1024]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_520: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_519, [1, -1, 16, 64]);  view_519 = None
    permute_276: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
    clone_197: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_521: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_124, [1, 128, 16, 64]);  mul_124 = None
    permute_277: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_521, [0, 2, 1, 3]);  view_521 = None
    clone_198: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_522: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_198, [16, -1, 64]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_523: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_196, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_524: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_197, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_278: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_523, [0, 2, 1]);  view_523 = None
    bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_522, permute_278);  view_522 = permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_525: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None
    add_153: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_525, expand_1);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_526: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_153, [16, 128, 128]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_26: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_526, [-1], True)
    sub_73: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_526, amax_26);  view_526 = amax_26 = None
    exp_26: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_27: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_26: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_199: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_53: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_199, view_524);  clone_199 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_527: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_53, [1, 16, 128, 64]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_279: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_200: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_528: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_200, [1, 128, 1024]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_529: "f32[128, 1024]" = torch.ops.aten.view.default(view_528, [128, 1024]);  view_528 = None
    permute_280: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
    addmm_145: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg387_1, view_529, permute_280);  arg387_1 = view_529 = permute_280 = None
    view_530: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_145, [1, 128, 1024]);  addmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_201: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_530);  view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_154: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_150, clone_201);  add_150 = clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    add_155: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_74: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_154, getitem_95);  getitem_95 = None
    mul_125: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_47);  sub_74 = rsqrt_47 = None
    mul_126: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_125, arg388_1);  mul_125 = arg388_1 = None
    add_156: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_126, arg389_1);  mul_126 = arg389_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_531: "f32[128, 1024]" = torch.ops.aten.view.default(add_156, [128, 1024]);  add_156 = None
    permute_281: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
    addmm_146: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg391_1, view_531, permute_281);  arg391_1 = view_531 = permute_281 = None
    view_532: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_146, [1, 128, 1024]);  addmm_146 = None
    mul_127: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_532, 0.125);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_533: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_282: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
    addmm_147: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg393_1, view_533, permute_282);  arg393_1 = view_533 = permute_282 = None
    view_534: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_147, [1, 128, 1024]);  addmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_535: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_534, [1, -1, 16, 64]);  view_534 = None
    permute_283: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
    clone_202: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_536: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_284: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg394_1, [1, 0]);  arg394_1 = None
    addmm_148: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg395_1, view_536, permute_284);  arg395_1 = view_536 = permute_284 = None
    view_537: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_148, [1, 128, 1024]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_538: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_537, [1, -1, 16, 64]);  view_537 = None
    permute_285: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
    clone_203: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_539: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_127, [1, 128, 16, 64]);  mul_127 = None
    permute_286: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
    clone_204: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_540: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_204, [16, -1, 64]);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_541: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_202, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_542: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_203, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_287: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_541, [0, 2, 1]);  view_541 = None
    bmm_54: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_540, permute_287);  view_540 = permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_27: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_54, [-1], True)
    sub_75: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_54, amax_27);  bmm_54 = amax_27 = None
    exp_27: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
    sum_28: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
    div_27: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_205: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_55: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_205, view_542);  clone_205 = view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_543: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_55, [1, 16, 128, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_288: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_206: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_544: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_206, [1, 128, 1024]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_545: "f32[128, 1024]" = torch.ops.aten.view.default(view_544, [128, 1024]);  view_544 = None
    permute_289: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
    addmm_149: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg397_1, view_545, permute_289);  arg397_1 = view_545 = permute_289 = None
    view_546: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_149, [1, 128, 1024]);  addmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_207: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_546);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_157: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_154, clone_207);  add_154 = clone_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    add_158: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_76: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_157, getitem_97);  getitem_97 = None
    mul_128: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_48);  sub_76 = rsqrt_48 = None
    mul_129: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_128, arg398_1);  mul_128 = arg398_1 = None
    add_159: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_129, arg399_1);  mul_129 = arg399_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_547: "f32[128, 1024]" = torch.ops.aten.view.default(add_159, [128, 1024]);  add_159 = None
    permute_290: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg400_1, [1, 0]);  arg400_1 = None
    addmm_150: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg401_1, view_547, permute_290);  arg401_1 = view_547 = permute_290 = None
    view_548: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_150, [1, 128, 4096]);  addmm_150 = None
    relu_19: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_548);  view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_208: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_19);  relu_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_549: "f32[128, 4096]" = torch.ops.aten.view.default(clone_208, [128, 4096]);  clone_208 = None
    permute_291: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg402_1, [1, 0]);  arg402_1 = None
    addmm_151: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg403_1, view_549, permute_291);  arg403_1 = view_549 = permute_291 = None
    view_550: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_151, [1, 128, 1024]);  addmm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_209: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_550);  view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_160: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_157, clone_209);  add_157 = clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 128, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 128, 1]" = var_mean_49[1];  var_mean_49 = None
    add_161: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_77: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_160, getitem_99);  getitem_99 = None
    mul_130: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_49);  sub_77 = rsqrt_49 = None
    mul_131: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_130, arg404_1);  mul_130 = arg404_1 = None
    add_162: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_131, arg405_1);  mul_131 = arg405_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_551: "f32[128, 1024]" = torch.ops.aten.view.default(add_162, [128, 1024])
    permute_292: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg406_1, [1, 0]);  arg406_1 = None
    addmm_152: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg407_1, view_551, permute_292);  arg407_1 = view_551 = permute_292 = None
    view_552: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_152, [1, 128, 1024]);  addmm_152 = None
    mul_132: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_552, 0.125);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_553: "f32[128, 1024]" = torch.ops.aten.view.default(add_162, [128, 1024])
    permute_293: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg408_1, [1, 0]);  arg408_1 = None
    addmm_153: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg409_1, view_553, permute_293);  arg409_1 = view_553 = permute_293 = None
    view_554: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_153, [1, 128, 1024]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_555: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_554, [1, -1, 16, 64]);  view_554 = None
    permute_294: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
    clone_210: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_556: "f32[128, 1024]" = torch.ops.aten.view.default(add_162, [128, 1024]);  add_162 = None
    permute_295: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg410_1, [1, 0]);  arg410_1 = None
    addmm_154: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg411_1, view_556, permute_295);  arg411_1 = view_556 = permute_295 = None
    view_557: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_154, [1, 128, 1024]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_558: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_557, [1, -1, 16, 64]);  view_557 = None
    permute_296: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
    clone_211: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_559: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_132, [1, 128, 16, 64]);  mul_132 = None
    permute_297: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None
    clone_212: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_560: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_212, [16, -1, 64]);  clone_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_561: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_210, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_562: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_211, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_298: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_561, [0, 2, 1]);  view_561 = None
    bmm_56: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_560, permute_298);  view_560 = permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_563: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_56, [1, 16, 128, 128]);  bmm_56 = None
    add_163: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_563, expand_1);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_564: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_163, [16, 128, 128]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_28: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_564, [-1], True)
    sub_78: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_564, amax_28);  view_564 = amax_28 = None
    exp_28: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_78);  sub_78 = None
    sum_29: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
    div_28: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_213: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_28);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_57: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_213, view_562);  clone_213 = view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_565: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_57, [1, 16, 128, 64]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_299: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_565, [0, 2, 1, 3]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_214: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_566: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_214, [1, 128, 1024]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_567: "f32[128, 1024]" = torch.ops.aten.view.default(view_566, [128, 1024]);  view_566 = None
    permute_300: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg412_1, [1, 0]);  arg412_1 = None
    addmm_155: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg413_1, view_567, permute_300);  arg413_1 = view_567 = permute_300 = None
    view_568: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_155, [1, 128, 1024]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_215: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_568);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_164: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_160, clone_215);  add_160 = clone_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_50 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 128, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 128, 1]" = var_mean_50[1];  var_mean_50 = None
    add_165: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_79: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_164, getitem_101);  getitem_101 = None
    mul_133: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_50);  sub_79 = rsqrt_50 = None
    mul_134: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_133, arg414_1);  mul_133 = arg414_1 = None
    add_166: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_134, arg415_1);  mul_134 = arg415_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_569: "f32[128, 1024]" = torch.ops.aten.view.default(add_166, [128, 1024]);  add_166 = None
    permute_301: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
    addmm_156: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg417_1, view_569, permute_301);  arg417_1 = view_569 = permute_301 = None
    view_570: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_156, [1, 128, 1024]);  addmm_156 = None
    mul_135: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_570, 0.125);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_571: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_302: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg418_1, [1, 0]);  arg418_1 = None
    addmm_157: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg419_1, view_571, permute_302);  arg419_1 = view_571 = permute_302 = None
    view_572: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_157, [1, 128, 1024]);  addmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_573: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_572, [1, -1, 16, 64]);  view_572 = None
    permute_303: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    clone_216: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_574: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_304: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg420_1, [1, 0]);  arg420_1 = None
    addmm_158: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg421_1, view_574, permute_304);  arg421_1 = view_574 = permute_304 = None
    view_575: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_158, [1, 128, 1024]);  addmm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_576: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_575, [1, -1, 16, 64]);  view_575 = None
    permute_305: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None
    clone_217: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_577: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_135, [1, 128, 16, 64]);  mul_135 = None
    permute_306: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None
    clone_218: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_578: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_218, [16, -1, 64]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_579: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_216, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_580: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_217, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_307: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None
    bmm_58: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_578, permute_307);  view_578 = permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_29: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_58, [-1], True)
    sub_80: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_58, amax_29);  bmm_58 = amax_29 = None
    exp_29: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
    sum_30: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
    div_29: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_219: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_59: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_219, view_580);  clone_219 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_581: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_59, [1, 16, 128, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_308: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_581, [0, 2, 1, 3]);  view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_220: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    view_582: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_220, [1, 128, 1024]);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_583: "f32[128, 1024]" = torch.ops.aten.view.default(view_582, [128, 1024]);  view_582 = None
    permute_309: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg422_1, [1, 0]);  arg422_1 = None
    addmm_159: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg423_1, view_583, permute_309);  arg423_1 = view_583 = permute_309 = None
    view_584: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_159, [1, 128, 1024]);  addmm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_221: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_584);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_167: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_164, clone_221);  add_164 = clone_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_51 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 128, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 128, 1]" = var_mean_51[1];  var_mean_51 = None
    add_168: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_81: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_167, getitem_103);  getitem_103 = None
    mul_136: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_51);  sub_81 = rsqrt_51 = None
    mul_137: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_136, arg424_1);  mul_136 = arg424_1 = None
    add_169: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_137, arg425_1);  mul_137 = arg425_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_585: "f32[128, 1024]" = torch.ops.aten.view.default(add_169, [128, 1024]);  add_169 = None
    permute_310: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg426_1, [1, 0]);  arg426_1 = None
    addmm_160: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg427_1, view_585, permute_310);  arg427_1 = view_585 = permute_310 = None
    view_586: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_160, [1, 128, 4096]);  addmm_160 = None
    relu_20: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_586);  view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_222: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_20);  relu_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_587: "f32[128, 4096]" = torch.ops.aten.view.default(clone_222, [128, 4096]);  clone_222 = None
    permute_311: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg428_1, [1, 0]);  arg428_1 = None
    addmm_161: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg429_1, view_587, permute_311);  arg429_1 = view_587 = permute_311 = None
    view_588: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_161, [1, 128, 1024]);  addmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_223: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_588);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_170: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_167, clone_223);  add_167 = clone_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_52 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 128, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 128, 1]" = var_mean_52[1];  var_mean_52 = None
    add_171: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_82: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_170, getitem_105);  getitem_105 = None
    mul_138: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_52);  sub_82 = rsqrt_52 = None
    mul_139: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_138, arg430_1);  mul_138 = arg430_1 = None
    add_172: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_139, arg431_1);  mul_139 = arg431_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_589: "f32[128, 1024]" = torch.ops.aten.view.default(add_172, [128, 1024])
    permute_312: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg432_1, [1, 0]);  arg432_1 = None
    addmm_162: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg433_1, view_589, permute_312);  arg433_1 = view_589 = permute_312 = None
    view_590: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_162, [1, 128, 1024]);  addmm_162 = None
    mul_140: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_590, 0.125);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_591: "f32[128, 1024]" = torch.ops.aten.view.default(add_172, [128, 1024])
    permute_313: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
    addmm_163: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg435_1, view_591, permute_313);  arg435_1 = view_591 = permute_313 = None
    view_592: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_163, [1, 128, 1024]);  addmm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_593: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_592, [1, -1, 16, 64]);  view_592 = None
    permute_314: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    clone_224: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_594: "f32[128, 1024]" = torch.ops.aten.view.default(add_172, [128, 1024]);  add_172 = None
    permute_315: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg436_1, [1, 0]);  arg436_1 = None
    addmm_164: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg437_1, view_594, permute_315);  arg437_1 = view_594 = permute_315 = None
    view_595: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_164, [1, 128, 1024]);  addmm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_596: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_595, [1, -1, 16, 64]);  view_595 = None
    permute_316: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
    clone_225: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_597: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_140, [1, 128, 16, 64]);  mul_140 = None
    permute_317: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_226: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_598: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_226, [16, -1, 64]);  clone_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_599: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_224, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_600: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_225, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_318: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_599, [0, 2, 1]);  view_599 = None
    bmm_60: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_598, permute_318);  view_598 = permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_601: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_60, [1, 16, 128, 128]);  bmm_60 = None
    add_173: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_601, expand_1);  view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_602: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_173, [16, 128, 128]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_30: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_602, [-1], True)
    sub_83: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_602, amax_30);  view_602 = amax_30 = None
    exp_30: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
    sum_31: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
    div_30: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_227: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_61: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_227, view_600);  clone_227 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_603: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_61, [1, 16, 128, 64]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_319: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_228: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_604: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_228, [1, 128, 1024]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_605: "f32[128, 1024]" = torch.ops.aten.view.default(view_604, [128, 1024]);  view_604 = None
    permute_320: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg438_1, [1, 0]);  arg438_1 = None
    addmm_165: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg439_1, view_605, permute_320);  arg439_1 = view_605 = permute_320 = None
    view_606: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_165, [1, 128, 1024]);  addmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_229: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_606);  view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_174: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_170, clone_229);  add_170 = clone_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_53 = torch.ops.aten.var_mean.correction(add_174, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 128, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 128, 1]" = var_mean_53[1];  var_mean_53 = None
    add_175: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_84: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_174, getitem_107);  getitem_107 = None
    mul_141: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_53);  sub_84 = rsqrt_53 = None
    mul_142: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_141, arg440_1);  mul_141 = arg440_1 = None
    add_176: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_142, arg441_1);  mul_142 = arg441_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_607: "f32[128, 1024]" = torch.ops.aten.view.default(add_176, [128, 1024]);  add_176 = None
    permute_321: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg442_1, [1, 0]);  arg442_1 = None
    addmm_166: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg443_1, view_607, permute_321);  arg443_1 = view_607 = permute_321 = None
    view_608: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_166, [1, 128, 1024]);  addmm_166 = None
    mul_143: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_608, 0.125);  view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_609: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_322: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg444_1, [1, 0]);  arg444_1 = None
    addmm_167: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg445_1, view_609, permute_322);  arg445_1 = view_609 = permute_322 = None
    view_610: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_167, [1, 128, 1024]);  addmm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_611: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_610, [1, -1, 16, 64]);  view_610 = None
    permute_323: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None
    clone_230: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_612: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_324: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg446_1, [1, 0]);  arg446_1 = None
    addmm_168: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg447_1, view_612, permute_324);  arg447_1 = view_612 = permute_324 = None
    view_613: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_168, [1, 128, 1024]);  addmm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_614: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_613, [1, -1, 16, 64]);  view_613 = None
    permute_325: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
    clone_231: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_615: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_143, [1, 128, 16, 64]);  mul_143 = None
    permute_326: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None
    clone_232: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_616: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_232, [16, -1, 64]);  clone_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_617: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_230, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_618: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_231, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_327: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_617, [0, 2, 1]);  view_617 = None
    bmm_62: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_616, permute_327);  view_616 = permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_31: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_62, [-1], True)
    sub_85: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_62, amax_31);  bmm_62 = amax_31 = None
    exp_31: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
    sum_32: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
    div_31: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_233: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_31);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_63: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_233, view_618);  clone_233 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_619: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_63, [1, 16, 128, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_328: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_234: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_620: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_234, [1, 128, 1024]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_621: "f32[128, 1024]" = torch.ops.aten.view.default(view_620, [128, 1024]);  view_620 = None
    permute_329: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg448_1, [1, 0]);  arg448_1 = None
    addmm_169: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg449_1, view_621, permute_329);  arg449_1 = view_621 = permute_329 = None
    view_622: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_169, [1, 128, 1024]);  addmm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_235: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_622);  view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_177: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_174, clone_235);  add_174 = clone_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_54 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 128, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 128, 1]" = var_mean_54[1];  var_mean_54 = None
    add_178: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_86: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_177, getitem_109);  getitem_109 = None
    mul_144: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_54);  sub_86 = rsqrt_54 = None
    mul_145: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_144, arg450_1);  mul_144 = arg450_1 = None
    add_179: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_145, arg451_1);  mul_145 = arg451_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_623: "f32[128, 1024]" = torch.ops.aten.view.default(add_179, [128, 1024]);  add_179 = None
    permute_330: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
    addmm_170: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg453_1, view_623, permute_330);  arg453_1 = view_623 = permute_330 = None
    view_624: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_170, [1, 128, 4096]);  addmm_170 = None
    relu_21: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_624);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_236: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_21);  relu_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_625: "f32[128, 4096]" = torch.ops.aten.view.default(clone_236, [128, 4096]);  clone_236 = None
    permute_331: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg454_1, [1, 0]);  arg454_1 = None
    addmm_171: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg455_1, view_625, permute_331);  arg455_1 = view_625 = permute_331 = None
    view_626: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_171, [1, 128, 1024]);  addmm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_237: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_626);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_180: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_177, clone_237);  add_177 = clone_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_55 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 128, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 128, 1]" = var_mean_55[1];  var_mean_55 = None
    add_181: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_87: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_180, getitem_111);  getitem_111 = None
    mul_146: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_55);  sub_87 = rsqrt_55 = None
    mul_147: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_146, arg456_1);  mul_146 = arg456_1 = None
    add_182: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_147, arg457_1);  mul_147 = arg457_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_627: "f32[128, 1024]" = torch.ops.aten.view.default(add_182, [128, 1024])
    permute_332: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg458_1, [1, 0]);  arg458_1 = None
    addmm_172: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg459_1, view_627, permute_332);  arg459_1 = view_627 = permute_332 = None
    view_628: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_172, [1, 128, 1024]);  addmm_172 = None
    mul_148: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_628, 0.125);  view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_629: "f32[128, 1024]" = torch.ops.aten.view.default(add_182, [128, 1024])
    permute_333: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg460_1, [1, 0]);  arg460_1 = None
    addmm_173: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg461_1, view_629, permute_333);  arg461_1 = view_629 = permute_333 = None
    view_630: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_173, [1, 128, 1024]);  addmm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_631: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_630, [1, -1, 16, 64]);  view_630 = None
    permute_334: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
    clone_238: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_632: "f32[128, 1024]" = torch.ops.aten.view.default(add_182, [128, 1024]);  add_182 = None
    permute_335: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg462_1, [1, 0]);  arg462_1 = None
    addmm_174: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg463_1, view_632, permute_335);  arg463_1 = view_632 = permute_335 = None
    view_633: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_174, [1, 128, 1024]);  addmm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_634: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_633, [1, -1, 16, 64]);  view_633 = None
    permute_336: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
    clone_239: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_635: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_148, [1, 128, 16, 64]);  mul_148 = None
    permute_337: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
    clone_240: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_636: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_240, [16, -1, 64]);  clone_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_637: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_238, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_638: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_239, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_338: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_637, [0, 2, 1]);  view_637 = None
    bmm_64: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_636, permute_338);  view_636 = permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_639: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_64, [1, 16, 128, 128]);  bmm_64 = None
    add_183: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_639, expand_1);  view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_640: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_183, [16, 128, 128]);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_32: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_640, [-1], True)
    sub_88: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_640, amax_32);  view_640 = amax_32 = None
    exp_32: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_88);  sub_88 = None
    sum_33: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
    div_32: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_241: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_65: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_241, view_638);  clone_241 = view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_641: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_65, [1, 16, 128, 64]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_339: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_641, [0, 2, 1, 3]);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_242: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_339, memory_format = torch.contiguous_format);  permute_339 = None
    view_642: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_242, [1, 128, 1024]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_643: "f32[128, 1024]" = torch.ops.aten.view.default(view_642, [128, 1024]);  view_642 = None
    permute_340: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg464_1, [1, 0]);  arg464_1 = None
    addmm_175: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg465_1, view_643, permute_340);  arg465_1 = view_643 = permute_340 = None
    view_644: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_175, [1, 128, 1024]);  addmm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_243: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_644);  view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_184: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_180, clone_243);  add_180 = clone_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_56 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 128, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 128, 1]" = var_mean_56[1];  var_mean_56 = None
    add_185: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_89: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_184, getitem_113);  getitem_113 = None
    mul_149: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_56);  sub_89 = rsqrt_56 = None
    mul_150: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_149, arg466_1);  mul_149 = arg466_1 = None
    add_186: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_150, arg467_1);  mul_150 = arg467_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_645: "f32[128, 1024]" = torch.ops.aten.view.default(add_186, [128, 1024]);  add_186 = None
    permute_341: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
    addmm_176: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg469_1, view_645, permute_341);  arg469_1 = view_645 = permute_341 = None
    view_646: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_176, [1, 128, 1024]);  addmm_176 = None
    mul_151: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_646, 0.125);  view_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_647: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_342: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
    addmm_177: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg471_1, view_647, permute_342);  arg471_1 = view_647 = permute_342 = None
    view_648: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_177, [1, 128, 1024]);  addmm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_649: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_648, [1, -1, 16, 64]);  view_648 = None
    permute_343: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
    clone_244: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_650: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_344: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg472_1, [1, 0]);  arg472_1 = None
    addmm_178: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg473_1, view_650, permute_344);  arg473_1 = view_650 = permute_344 = None
    view_651: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_178, [1, 128, 1024]);  addmm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_652: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_651, [1, -1, 16, 64]);  view_651 = None
    permute_345: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
    clone_245: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_653: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_151, [1, 128, 16, 64]);  mul_151 = None
    permute_346: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    clone_246: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_654: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_246, [16, -1, 64]);  clone_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_655: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_244, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_656: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_245, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_347: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
    bmm_66: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_654, permute_347);  view_654 = permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_33: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_66, [-1], True)
    sub_90: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_66, amax_33);  bmm_66 = amax_33 = None
    exp_33: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_90);  sub_90 = None
    sum_34: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
    div_33: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_247: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_33);  div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_67: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_247, view_656);  clone_247 = view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_657: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_67, [1, 16, 128, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_348: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_248: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_658: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_248, [1, 128, 1024]);  clone_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_659: "f32[128, 1024]" = torch.ops.aten.view.default(view_658, [128, 1024]);  view_658 = None
    permute_349: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg474_1, [1, 0]);  arg474_1 = None
    addmm_179: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg475_1, view_659, permute_349);  arg475_1 = view_659 = permute_349 = None
    view_660: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_179, [1, 128, 1024]);  addmm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_249: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_660);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_187: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_184, clone_249);  add_184 = clone_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_57 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
    getitem_114: "f32[1, 128, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 128, 1]" = var_mean_57[1];  var_mean_57 = None
    add_188: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_57: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_91: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_187, getitem_115);  getitem_115 = None
    mul_152: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_57);  sub_91 = rsqrt_57 = None
    mul_153: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_152, arg476_1);  mul_152 = arg476_1 = None
    add_189: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_153, arg477_1);  mul_153 = arg477_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_661: "f32[128, 1024]" = torch.ops.aten.view.default(add_189, [128, 1024]);  add_189 = None
    permute_350: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg478_1, [1, 0]);  arg478_1 = None
    addmm_180: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg479_1, view_661, permute_350);  arg479_1 = view_661 = permute_350 = None
    view_662: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_180, [1, 128, 4096]);  addmm_180 = None
    relu_22: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_662);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_250: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_22);  relu_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_663: "f32[128, 4096]" = torch.ops.aten.view.default(clone_250, [128, 4096]);  clone_250 = None
    permute_351: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg480_1, [1, 0]);  arg480_1 = None
    addmm_181: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg481_1, view_663, permute_351);  arg481_1 = view_663 = permute_351 = None
    view_664: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_181, [1, 128, 1024]);  addmm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_251: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_664);  view_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_190: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_187, clone_251);  add_187 = clone_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_58 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
    getitem_116: "f32[1, 128, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 128, 1]" = var_mean_58[1];  var_mean_58 = None
    add_191: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_58: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_92: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_190, getitem_117);  getitem_117 = None
    mul_154: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_58);  sub_92 = rsqrt_58 = None
    mul_155: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_154, arg482_1);  mul_154 = arg482_1 = None
    add_192: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_155, arg483_1);  mul_155 = arg483_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_665: "f32[128, 1024]" = torch.ops.aten.view.default(add_192, [128, 1024])
    permute_352: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg484_1, [1, 0]);  arg484_1 = None
    addmm_182: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg485_1, view_665, permute_352);  arg485_1 = view_665 = permute_352 = None
    view_666: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_182, [1, 128, 1024]);  addmm_182 = None
    mul_156: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_666, 0.125);  view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_667: "f32[128, 1024]" = torch.ops.aten.view.default(add_192, [128, 1024])
    permute_353: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg486_1, [1, 0]);  arg486_1 = None
    addmm_183: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg487_1, view_667, permute_353);  arg487_1 = view_667 = permute_353 = None
    view_668: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_183, [1, 128, 1024]);  addmm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_669: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_668, [1, -1, 16, 64]);  view_668 = None
    permute_354: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_669, [0, 2, 1, 3]);  view_669 = None
    clone_252: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_670: "f32[128, 1024]" = torch.ops.aten.view.default(add_192, [128, 1024]);  add_192 = None
    permute_355: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg488_1, [1, 0]);  arg488_1 = None
    addmm_184: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg489_1, view_670, permute_355);  arg489_1 = view_670 = permute_355 = None
    view_671: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_184, [1, 128, 1024]);  addmm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_672: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_671, [1, -1, 16, 64]);  view_671 = None
    permute_356: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
    clone_253: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_673: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_156, [1, 128, 16, 64]);  mul_156 = None
    permute_357: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
    clone_254: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_674: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_254, [16, -1, 64]);  clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_675: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_252, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_676: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_253, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_358: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_675, [0, 2, 1]);  view_675 = None
    bmm_68: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_674, permute_358);  view_674 = permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_677: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_68, [1, 16, 128, 128]);  bmm_68 = None
    add_193: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_677, expand_1);  view_677 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_678: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_193, [16, 128, 128]);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_34: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_678, [-1], True)
    sub_93: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_678, amax_34);  view_678 = amax_34 = None
    exp_34: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
    sum_35: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
    div_34: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_255: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_34);  div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_69: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_255, view_676);  clone_255 = view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_679: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_69, [1, 16, 128, 64]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_359: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_256: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_680: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_256, [1, 128, 1024]);  clone_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_681: "f32[128, 1024]" = torch.ops.aten.view.default(view_680, [128, 1024]);  view_680 = None
    permute_360: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg490_1, [1, 0]);  arg490_1 = None
    addmm_185: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg491_1, view_681, permute_360);  arg491_1 = view_681 = permute_360 = None
    view_682: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_185, [1, 128, 1024]);  addmm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_257: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_682);  view_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_194: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_190, clone_257);  add_190 = clone_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_59 = torch.ops.aten.var_mean.correction(add_194, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 128, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 128, 1]" = var_mean_59[1];  var_mean_59 = None
    add_195: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    sub_94: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_194, getitem_119);  getitem_119 = None
    mul_157: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_59);  sub_94 = rsqrt_59 = None
    mul_158: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_157, arg492_1);  mul_157 = arg492_1 = None
    add_196: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_158, arg493_1);  mul_158 = arg493_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_683: "f32[128, 1024]" = torch.ops.aten.view.default(add_196, [128, 1024]);  add_196 = None
    permute_361: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg494_1, [1, 0]);  arg494_1 = None
    addmm_186: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg495_1, view_683, permute_361);  arg495_1 = view_683 = permute_361 = None
    view_684: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_186, [1, 128, 1024]);  addmm_186 = None
    mul_159: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_684, 0.125);  view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_685: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_362: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg496_1, [1, 0]);  arg496_1 = None
    addmm_187: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg497_1, view_685, permute_362);  arg497_1 = view_685 = permute_362 = None
    view_686: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_187, [1, 128, 1024]);  addmm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_687: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_686, [1, -1, 16, 64]);  view_686 = None
    permute_363: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_687, [0, 2, 1, 3]);  view_687 = None
    clone_258: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_688: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_364: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg498_1, [1, 0]);  arg498_1 = None
    addmm_188: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg499_1, view_688, permute_364);  arg499_1 = view_688 = permute_364 = None
    view_689: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_188, [1, 128, 1024]);  addmm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_690: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_689, [1, -1, 16, 64]);  view_689 = None
    permute_365: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
    clone_259: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_691: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_159, [1, 128, 16, 64]);  mul_159 = None
    permute_366: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None
    clone_260: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_692: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_260, [16, -1, 64]);  clone_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_693: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_258, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_694: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_259, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_367: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_693, [0, 2, 1]);  view_693 = None
    bmm_70: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_692, permute_367);  view_692 = permute_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_35: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_70, [-1], True)
    sub_95: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_70, amax_35);  bmm_70 = amax_35 = None
    exp_35: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_95);  sub_95 = None
    sum_36: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
    div_35: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_261: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_71: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_261, view_694);  clone_261 = view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_695: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_71, [1, 16, 128, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_368: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_262: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_696: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_262, [1, 128, 1024]);  clone_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_697: "f32[128, 1024]" = torch.ops.aten.view.default(view_696, [128, 1024]);  view_696 = None
    permute_369: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg500_1, [1, 0]);  arg500_1 = None
    addmm_189: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg501_1, view_697, permute_369);  arg501_1 = view_697 = permute_369 = None
    view_698: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_189, [1, 128, 1024]);  addmm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_263: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_698);  view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_197: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_194, clone_263);  add_194 = clone_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_60 = torch.ops.aten.var_mean.correction(add_197, [2], correction = 0, keepdim = True)
    getitem_120: "f32[1, 128, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 128, 1]" = var_mean_60[1];  var_mean_60 = None
    add_198: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_60: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_96: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_197, getitem_121);  getitem_121 = None
    mul_160: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_60);  sub_96 = rsqrt_60 = None
    mul_161: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_160, arg502_1);  mul_160 = arg502_1 = None
    add_199: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_161, arg503_1);  mul_161 = arg503_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_699: "f32[128, 1024]" = torch.ops.aten.view.default(add_199, [128, 1024]);  add_199 = None
    permute_370: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
    addmm_190: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg505_1, view_699, permute_370);  arg505_1 = view_699 = permute_370 = None
    view_700: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_190, [1, 128, 4096]);  addmm_190 = None
    relu_23: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_700);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_264: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(relu_23);  relu_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_701: "f32[128, 4096]" = torch.ops.aten.view.default(clone_264, [128, 4096]);  clone_264 = None
    permute_371: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg506_1, [1, 0]);  arg506_1 = None
    addmm_191: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg507_1, view_701, permute_371);  arg507_1 = view_701 = permute_371 = None
    view_702: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_191, [1, 128, 1024]);  addmm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_265: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_702);  view_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_200: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_197, clone_265);  add_197 = clone_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1114, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_61 = torch.ops.aten.var_mean.correction(add_200, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 128, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 128, 1]" = var_mean_61[1];  var_mean_61 = None
    add_201: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_61: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    sub_97: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_200, getitem_123);  add_200 = getitem_123 = None
    mul_162: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_61);  sub_97 = rsqrt_61 = None
    mul_163: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_162, arg508_1);  mul_162 = arg508_1 = None
    add_202: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_163, arg509_1);  mul_163 = arg509_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1331, code: lm_logits = self.lm_head(outputs[0])
    permute_372: "f32[1024, 128112]" = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
    view_703: "f32[128, 1024]" = torch.ops.aten.view.default(add_202, [128, 1024]);  add_202 = None
    mm: "f32[128, 128112]" = torch.ops.aten.mm.default(view_703, permute_372);  view_703 = permute_372 = None
    view_704: "f32[1, 128, 128112]" = torch.ops.aten.view.default(mm, [1, 128, 128112]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1338, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_705: "f32[128, 128112]" = torch.ops.aten.view.default(view_704, [-1, 128112])
    view_706: "i64[128]" = torch.ops.aten.view.default(arg513_1, [-1]);  arg513_1 = None
    amax_36: "f32[128, 1]" = torch.ops.aten.amax.default(view_705, [1], True)
    sub_98: "f32[128, 128112]" = torch.ops.aten.sub.Tensor(view_705, amax_36);  view_705 = amax_36 = None
    exp_36: "f32[128, 128112]" = torch.ops.aten.exp.default(sub_98)
    sum_37: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [1], True);  exp_36 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_37);  sum_37 = None
    sub_99: "f32[128, 128112]" = torch.ops.aten.sub.Tensor(sub_98, log);  sub_98 = log = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_706, -100)
    scalar_tensor_1: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_1: "i64[128]" = torch.ops.aten.where.self(ne_2, view_706, scalar_tensor_1);  ne_2 = scalar_tensor_1 = None
    unsqueeze_4: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_99, 1, unsqueeze_4);  sub_99 = unsqueeze_4 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_3: "b8[128]" = torch.ops.aten.ne.Scalar(view_706, -100)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[128]" = torch.ops.aten.where.self(ne_3, neg, scalar_tensor_2);  ne_3 = neg = scalar_tensor_2 = None
    ne_4: "b8[128]" = torch.ops.aten.ne.Scalar(view_706, -100);  view_706 = None
    sum_38: "i64[]" = torch.ops.aten.sum.default(ne_4);  ne_4 = None
    convert_element_type_6: "f32[]" = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
    sum_39: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    div_36: "f32[]" = torch.ops.aten.div.Tensor(sum_39, convert_element_type_6);  sum_39 = convert_element_type_6 = None
    return (div_36, view_704, clone_98, clone_99, clone_104, clone_105, clone_112, clone_113, clone_118, clone_119, clone_126, clone_127, clone_132, clone_133, clone_140, clone_141, clone_146, clone_147, clone_154, clone_155, clone_160, clone_161, clone_168, clone_169, clone_174, clone_175, clone_182, clone_183, clone_188, clone_189, clone_196, clone_197, clone_202, clone_203, clone_210, clone_211, clone_216, clone_217, clone_224, clone_225, clone_230, clone_231, clone_238, clone_239, clone_244, clone_245, clone_252, clone_253, clone_258, clone_259, add_76)
    