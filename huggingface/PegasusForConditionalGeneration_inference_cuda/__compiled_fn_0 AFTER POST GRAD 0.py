from __future__ import annotations



def forward(self, arg0_1: "f32[1024, 1024]", arg1_1: "f32[1024, 1024]", arg2_1: "f32[50265, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024, 1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024]", arg14_1: "f32[1024]", arg15_1: "f32[4096, 1024]", arg16_1: "f32[4096]", arg17_1: "f32[1024, 4096]", arg18_1: "f32[1024]", arg19_1: "f32[1024]", arg20_1: "f32[1024]", arg21_1: "f32[1024, 1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024]", arg30_1: "f32[1024]", arg31_1: "f32[4096, 1024]", arg32_1: "f32[4096]", arg33_1: "f32[1024, 4096]", arg34_1: "f32[1024]", arg35_1: "f32[1024]", arg36_1: "f32[1024]", arg37_1: "f32[1024, 1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024, 1024]", arg44_1: "f32[1024]", arg45_1: "f32[1024]", arg46_1: "f32[1024]", arg47_1: "f32[4096, 1024]", arg48_1: "f32[4096]", arg49_1: "f32[1024, 4096]", arg50_1: "f32[1024]", arg51_1: "f32[1024]", arg52_1: "f32[1024]", arg53_1: "f32[1024, 1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024, 1024]", arg60_1: "f32[1024]", arg61_1: "f32[1024]", arg62_1: "f32[1024]", arg63_1: "f32[4096, 1024]", arg64_1: "f32[4096]", arg65_1: "f32[1024, 4096]", arg66_1: "f32[1024]", arg67_1: "f32[1024]", arg68_1: "f32[1024]", arg69_1: "f32[1024, 1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024, 1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024]", arg78_1: "f32[1024]", arg79_1: "f32[4096, 1024]", arg80_1: "f32[4096]", arg81_1: "f32[1024, 4096]", arg82_1: "f32[1024]", arg83_1: "f32[1024]", arg84_1: "f32[1024]", arg85_1: "f32[1024, 1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024, 1024]", arg92_1: "f32[1024]", arg93_1: "f32[1024]", arg94_1: "f32[1024]", arg95_1: "f32[4096, 1024]", arg96_1: "f32[4096]", arg97_1: "f32[1024, 4096]", arg98_1: "f32[1024]", arg99_1: "f32[1024]", arg100_1: "f32[1024]", arg101_1: "f32[1024, 1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024, 1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024]", arg110_1: "f32[1024]", arg111_1: "f32[4096, 1024]", arg112_1: "f32[4096]", arg113_1: "f32[1024, 4096]", arg114_1: "f32[1024]", arg115_1: "f32[1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024, 1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024, 1024]", arg124_1: "f32[1024]", arg125_1: "f32[1024]", arg126_1: "f32[1024]", arg127_1: "f32[4096, 1024]", arg128_1: "f32[4096]", arg129_1: "f32[1024, 4096]", arg130_1: "f32[1024]", arg131_1: "f32[1024]", arg132_1: "f32[1024]", arg133_1: "f32[1024, 1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024, 1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024]", arg142_1: "f32[1024]", arg143_1: "f32[4096, 1024]", arg144_1: "f32[4096]", arg145_1: "f32[1024, 4096]", arg146_1: "f32[1024]", arg147_1: "f32[1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024, 1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024, 1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024]", arg158_1: "f32[1024]", arg159_1: "f32[4096, 1024]", arg160_1: "f32[4096]", arg161_1: "f32[1024, 4096]", arg162_1: "f32[1024]", arg163_1: "f32[1024]", arg164_1: "f32[1024]", arg165_1: "f32[1024, 1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024, 1024]", arg172_1: "f32[1024]", arg173_1: "f32[1024]", arg174_1: "f32[1024]", arg175_1: "f32[4096, 1024]", arg176_1: "f32[4096]", arg177_1: "f32[1024, 4096]", arg178_1: "f32[1024]", arg179_1: "f32[1024]", arg180_1: "f32[1024]", arg181_1: "f32[1024, 1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024, 1024]", arg188_1: "f32[1024]", arg189_1: "f32[1024]", arg190_1: "f32[1024]", arg191_1: "f32[4096, 1024]", arg192_1: "f32[4096]", arg193_1: "f32[1024, 4096]", arg194_1: "f32[1024]", arg195_1: "f32[1024]", arg196_1: "f32[1024]", arg197_1: "f32[1024]", arg198_1: "f32[1024]", arg199_1: "f32[1024, 1024]", arg200_1: "f32[1024]", arg201_1: "f32[1024, 1024]", arg202_1: "f32[1024]", arg203_1: "f32[1024, 1024]", arg204_1: "f32[1024]", arg205_1: "f32[1024, 1024]", arg206_1: "f32[1024]", arg207_1: "f32[1024]", arg208_1: "f32[1024]", arg209_1: "f32[1024, 1024]", arg210_1: "f32[1024]", arg211_1: "f32[1024, 1024]", arg212_1: "f32[1024]", arg213_1: "f32[1024, 1024]", arg214_1: "f32[1024]", arg215_1: "f32[1024, 1024]", arg216_1: "f32[1024]", arg217_1: "f32[1024]", arg218_1: "f32[1024]", arg219_1: "f32[4096, 1024]", arg220_1: "f32[4096]", arg221_1: "f32[1024, 4096]", arg222_1: "f32[1024]", arg223_1: "f32[1024]", arg224_1: "f32[1024]", arg225_1: "f32[1024, 1024]", arg226_1: "f32[1024]", arg227_1: "f32[1024, 1024]", arg228_1: "f32[1024]", arg229_1: "f32[1024, 1024]", arg230_1: "f32[1024]", arg231_1: "f32[1024, 1024]", arg232_1: "f32[1024]", arg233_1: "f32[1024]", arg234_1: "f32[1024]", arg235_1: "f32[1024, 1024]", arg236_1: "f32[1024]", arg237_1: "f32[1024, 1024]", arg238_1: "f32[1024]", arg239_1: "f32[1024, 1024]", arg240_1: "f32[1024]", arg241_1: "f32[1024, 1024]", arg242_1: "f32[1024]", arg243_1: "f32[1024]", arg244_1: "f32[1024]", arg245_1: "f32[4096, 1024]", arg246_1: "f32[4096]", arg247_1: "f32[1024, 4096]", arg248_1: "f32[1024]", arg249_1: "f32[1024]", arg250_1: "f32[1024]", arg251_1: "f32[1024, 1024]", arg252_1: "f32[1024]", arg253_1: "f32[1024, 1024]", arg254_1: "f32[1024]", arg255_1: "f32[1024, 1024]", arg256_1: "f32[1024]", arg257_1: "f32[1024, 1024]", arg258_1: "f32[1024]", arg259_1: "f32[1024]", arg260_1: "f32[1024]", arg261_1: "f32[1024, 1024]", arg262_1: "f32[1024]", arg263_1: "f32[1024, 1024]", arg264_1: "f32[1024]", arg265_1: "f32[1024, 1024]", arg266_1: "f32[1024]", arg267_1: "f32[1024, 1024]", arg268_1: "f32[1024]", arg269_1: "f32[1024]", arg270_1: "f32[1024]", arg271_1: "f32[4096, 1024]", arg272_1: "f32[4096]", arg273_1: "f32[1024, 4096]", arg274_1: "f32[1024]", arg275_1: "f32[1024]", arg276_1: "f32[1024]", arg277_1: "f32[1024, 1024]", arg278_1: "f32[1024]", arg279_1: "f32[1024, 1024]", arg280_1: "f32[1024]", arg281_1: "f32[1024, 1024]", arg282_1: "f32[1024]", arg283_1: "f32[1024, 1024]", arg284_1: "f32[1024]", arg285_1: "f32[1024]", arg286_1: "f32[1024]", arg287_1: "f32[1024, 1024]", arg288_1: "f32[1024]", arg289_1: "f32[1024, 1024]", arg290_1: "f32[1024]", arg291_1: "f32[1024, 1024]", arg292_1: "f32[1024]", arg293_1: "f32[1024, 1024]", arg294_1: "f32[1024]", arg295_1: "f32[1024]", arg296_1: "f32[1024]", arg297_1: "f32[4096, 1024]", arg298_1: "f32[4096]", arg299_1: "f32[1024, 4096]", arg300_1: "f32[1024]", arg301_1: "f32[1024]", arg302_1: "f32[1024]", arg303_1: "f32[1024, 1024]", arg304_1: "f32[1024]", arg305_1: "f32[1024, 1024]", arg306_1: "f32[1024]", arg307_1: "f32[1024, 1024]", arg308_1: "f32[1024]", arg309_1: "f32[1024, 1024]", arg310_1: "f32[1024]", arg311_1: "f32[1024]", arg312_1: "f32[1024]", arg313_1: "f32[1024, 1024]", arg314_1: "f32[1024]", arg315_1: "f32[1024, 1024]", arg316_1: "f32[1024]", arg317_1: "f32[1024, 1024]", arg318_1: "f32[1024]", arg319_1: "f32[1024, 1024]", arg320_1: "f32[1024]", arg321_1: "f32[1024]", arg322_1: "f32[1024]", arg323_1: "f32[4096, 1024]", arg324_1: "f32[4096]", arg325_1: "f32[1024, 4096]", arg326_1: "f32[1024]", arg327_1: "f32[1024]", arg328_1: "f32[1024]", arg329_1: "f32[1024, 1024]", arg330_1: "f32[1024]", arg331_1: "f32[1024, 1024]", arg332_1: "f32[1024]", arg333_1: "f32[1024, 1024]", arg334_1: "f32[1024]", arg335_1: "f32[1024, 1024]", arg336_1: "f32[1024]", arg337_1: "f32[1024]", arg338_1: "f32[1024]", arg339_1: "f32[1024, 1024]", arg340_1: "f32[1024]", arg341_1: "f32[1024, 1024]", arg342_1: "f32[1024]", arg343_1: "f32[1024, 1024]", arg344_1: "f32[1024]", arg345_1: "f32[1024, 1024]", arg346_1: "f32[1024]", arg347_1: "f32[1024]", arg348_1: "f32[1024]", arg349_1: "f32[4096, 1024]", arg350_1: "f32[4096]", arg351_1: "f32[1024, 4096]", arg352_1: "f32[1024]", arg353_1: "f32[1024]", arg354_1: "f32[1024]", arg355_1: "f32[1024, 1024]", arg356_1: "f32[1024]", arg357_1: "f32[1024, 1024]", arg358_1: "f32[1024]", arg359_1: "f32[1024, 1024]", arg360_1: "f32[1024]", arg361_1: "f32[1024, 1024]", arg362_1: "f32[1024]", arg363_1: "f32[1024]", arg364_1: "f32[1024]", arg365_1: "f32[1024, 1024]", arg366_1: "f32[1024]", arg367_1: "f32[1024, 1024]", arg368_1: "f32[1024]", arg369_1: "f32[1024, 1024]", arg370_1: "f32[1024]", arg371_1: "f32[1024, 1024]", arg372_1: "f32[1024]", arg373_1: "f32[1024]", arg374_1: "f32[1024]", arg375_1: "f32[4096, 1024]", arg376_1: "f32[4096]", arg377_1: "f32[1024, 4096]", arg378_1: "f32[1024]", arg379_1: "f32[1024]", arg380_1: "f32[1024]", arg381_1: "f32[1024, 1024]", arg382_1: "f32[1024]", arg383_1: "f32[1024, 1024]", arg384_1: "f32[1024]", arg385_1: "f32[1024, 1024]", arg386_1: "f32[1024]", arg387_1: "f32[1024, 1024]", arg388_1: "f32[1024]", arg389_1: "f32[1024]", arg390_1: "f32[1024]", arg391_1: "f32[1024, 1024]", arg392_1: "f32[1024]", arg393_1: "f32[1024, 1024]", arg394_1: "f32[1024]", arg395_1: "f32[1024, 1024]", arg396_1: "f32[1024]", arg397_1: "f32[1024, 1024]", arg398_1: "f32[1024]", arg399_1: "f32[1024]", arg400_1: "f32[1024]", arg401_1: "f32[4096, 1024]", arg402_1: "f32[4096]", arg403_1: "f32[1024, 4096]", arg404_1: "f32[1024]", arg405_1: "f32[1024]", arg406_1: "f32[1024]", arg407_1: "f32[1024, 1024]", arg408_1: "f32[1024]", arg409_1: "f32[1024, 1024]", arg410_1: "f32[1024]", arg411_1: "f32[1024, 1024]", arg412_1: "f32[1024]", arg413_1: "f32[1024, 1024]", arg414_1: "f32[1024]", arg415_1: "f32[1024]", arg416_1: "f32[1024]", arg417_1: "f32[1024, 1024]", arg418_1: "f32[1024]", arg419_1: "f32[1024, 1024]", arg420_1: "f32[1024]", arg421_1: "f32[1024, 1024]", arg422_1: "f32[1024]", arg423_1: "f32[1024, 1024]", arg424_1: "f32[1024]", arg425_1: "f32[1024]", arg426_1: "f32[1024]", arg427_1: "f32[4096, 1024]", arg428_1: "f32[4096]", arg429_1: "f32[1024, 4096]", arg430_1: "f32[1024]", arg431_1: "f32[1024]", arg432_1: "f32[1024]", arg433_1: "f32[1024, 1024]", arg434_1: "f32[1024]", arg435_1: "f32[1024, 1024]", arg436_1: "f32[1024]", arg437_1: "f32[1024, 1024]", arg438_1: "f32[1024]", arg439_1: "f32[1024, 1024]", arg440_1: "f32[1024]", arg441_1: "f32[1024]", arg442_1: "f32[1024]", arg443_1: "f32[1024, 1024]", arg444_1: "f32[1024]", arg445_1: "f32[1024, 1024]", arg446_1: "f32[1024]", arg447_1: "f32[1024, 1024]", arg448_1: "f32[1024]", arg449_1: "f32[1024, 1024]", arg450_1: "f32[1024]", arg451_1: "f32[1024]", arg452_1: "f32[1024]", arg453_1: "f32[4096, 1024]", arg454_1: "f32[4096]", arg455_1: "f32[1024, 4096]", arg456_1: "f32[1024]", arg457_1: "f32[1024]", arg458_1: "f32[1024]", arg459_1: "f32[1024, 1024]", arg460_1: "f32[1024]", arg461_1: "f32[1024, 1024]", arg462_1: "f32[1024]", arg463_1: "f32[1024, 1024]", arg464_1: "f32[1024]", arg465_1: "f32[1024, 1024]", arg466_1: "f32[1024]", arg467_1: "f32[1024]", arg468_1: "f32[1024]", arg469_1: "f32[1024, 1024]", arg470_1: "f32[1024]", arg471_1: "f32[1024, 1024]", arg472_1: "f32[1024]", arg473_1: "f32[1024, 1024]", arg474_1: "f32[1024]", arg475_1: "f32[1024, 1024]", arg476_1: "f32[1024]", arg477_1: "f32[1024]", arg478_1: "f32[1024]", arg479_1: "f32[4096, 1024]", arg480_1: "f32[4096]", arg481_1: "f32[1024, 4096]", arg482_1: "f32[1024]", arg483_1: "f32[1024]", arg484_1: "f32[1024]", arg485_1: "f32[1024, 1024]", arg486_1: "f32[1024]", arg487_1: "f32[1024, 1024]", arg488_1: "f32[1024]", arg489_1: "f32[1024, 1024]", arg490_1: "f32[1024]", arg491_1: "f32[1024, 1024]", arg492_1: "f32[1024]", arg493_1: "f32[1024]", arg494_1: "f32[1024]", arg495_1: "f32[1024, 1024]", arg496_1: "f32[1024]", arg497_1: "f32[1024, 1024]", arg498_1: "f32[1024]", arg499_1: "f32[1024, 1024]", arg500_1: "f32[1024]", arg501_1: "f32[1024, 1024]", arg502_1: "f32[1024]", arg503_1: "f32[1024]", arg504_1: "f32[1024]", arg505_1: "f32[4096, 1024]", arg506_1: "f32[4096]", arg507_1: "f32[1024, 4096]", arg508_1: "f32[1024]", arg509_1: "f32[1024]", arg510_1: "f32[1024]", arg511_1: "f32[50265, 1024]", arg512_1: "f32[1, 50265]", arg513_1: "i64[1, 128]", arg514_1: "i64[1, 128]", arg515_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:762, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(arg515_1, [-1, 128]);  arg515_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:769, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 128, 1024]" = torch.ops.aten.embedding.default(arg2_1, view, 0);  view = None
    mul: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:137, code: positions = torch.arange(
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[128, 1024]" = torch.ops.aten.embedding.default(arg0_1, iota);  arg0_1 = iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:773, code: hidden_states = inputs_embeds + embed_pos
    add: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    sub: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul_1: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_2: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_2, [128, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    
    # No stacktrace found for following nodes
    mm_default_191: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
    add_tensor_191: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_191, arg6_1);  mm_default_191 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_191, [1, 128, 1024]);  add_tensor_191 = None
    mul_3: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_3, [1, 128, 16, 64]);  mul_3 = None
    permute_5: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_3: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_10: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_3, [16, -1, 64]);  clone_3 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_69: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_2, [128, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    
    # No stacktrace found for following nodes
    mm_default_190: "f32[128, 1024]" = torch.ops.aten.mm.default(view_3, permute_1);  view_3 = permute_1 = None
    add_tensor_190: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_190, arg8_1);  mm_default_190 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_190, [1, 128, 1024]);  add_tensor_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_4, [1, -1, 16, 64]);  view_4 = None
    permute_2: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    clone_1: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_11: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_1, [16, -1, 64]);  clone_1 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_70: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_2, [128, 1024]);  add_2 = None
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    
    # No stacktrace found for following nodes
    mm_default_189: "f32[128, 1024]" = torch.ops.aten.mm.default(view_6, permute_3);  view_6 = permute_3 = None
    add_tensor_189: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_189, arg10_1);  mm_default_189 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_189, [1, 128, 1024]);  add_tensor_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_7, [1, -1, 16, 64]);  view_7 = None
    permute_4: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_12: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_2, [16, -1, 64]);  clone_2 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_71: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
    _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_69, unsqueeze_default_70, unsqueeze_default_71, None, True, scale = 1.0);  unsqueeze_default_69 = unsqueeze_default_70 = unsqueeze_default_71 = None
    getitem_147: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_23[0];  _scaled_dot_product_efficient_attention_default_23 = None
    squeeze_dim_23: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_147, 0);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_13: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_23, [1, 16, 128, 64]);  squeeze_dim_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_14: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_5, [1, 128, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_15: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_14, [128, 1024]);  view_14 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    
    # No stacktrace found for following nodes
    mm_default_188: "f32[128, 1024]" = torch.ops.aten.mm.default(view_15, permute_8);  view_15 = permute_8 = None
    add_tensor_188: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_188, arg12_1);  mm_default_188 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_188, [1, 128, 1024]);  add_tensor_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_3: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add, view_16);  add = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  getitem_3 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_4: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
    add_5: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_5, [128, 1024]);  add_5 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    
    # No stacktrace found for following nodes
    mm_default_187: "f32[128, 4096]" = torch.ops.aten.mm.default(view_17, permute_9);  view_17 = permute_9 = None
    add_tensor_187: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_187, arg16_1);  mm_default_187 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_18: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_187, [1, 128, 4096]);  add_tensor_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_6: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_19: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_8, [128, 4096]);  mul_8 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    
    # No stacktrace found for following nodes
    mm_default_186: "f32[128, 1024]" = torch.ops.aten.mm.default(view_19, permute_10);  view_19 = permute_10 = None
    add_tensor_186: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_186, arg18_1);  mm_default_186 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_186, [1, 128, 1024]);  add_tensor_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_7: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, view_20);  add_3 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  getitem_5 = None
    add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    mul_9: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
    add_9: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_21: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_9, [128, 1024])
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    
    # No stacktrace found for following nodes
    mm_default_185: "f32[128, 1024]" = torch.ops.aten.mm.default(view_21, permute_11);  view_21 = permute_11 = None
    add_tensor_185: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_185, arg22_1);  mm_default_185 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_22: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_185, [1, 128, 1024]);  add_tensor_185 = None
    mul_11: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_22, 0.125);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_11, [1, 128, 16, 64]);  mul_11 = None
    permute_16: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_11: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_30: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_11, [16, -1, 64]);  clone_11 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_66: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_23: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_9, [128, 1024])
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    
    # No stacktrace found for following nodes
    mm_default_184: "f32[128, 1024]" = torch.ops.aten.mm.default(view_23, permute_12);  view_23 = permute_12 = None
    add_tensor_184: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_184, arg24_1);  mm_default_184 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_184, [1, 128, 1024]);  add_tensor_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_24, [1, -1, 16, 64]);  view_24 = None
    permute_13: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_9: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_31: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_9, [16, -1, 64]);  clone_9 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_67: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_26: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_9, [128, 1024]);  add_9 = None
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    
    # No stacktrace found for following nodes
    mm_default_183: "f32[128, 1024]" = torch.ops.aten.mm.default(view_26, permute_14);  view_26 = permute_14 = None
    add_tensor_183: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_183, arg26_1);  mm_default_183 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_27: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_183, [1, 128, 1024]);  add_tensor_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_27, [1, -1, 16, 64]);  view_27 = None
    permute_15: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_10: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_32: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_10, [16, -1, 64]);  clone_10 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_68: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
    _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_66, unsqueeze_default_67, unsqueeze_default_68, None, True, scale = 1.0);  unsqueeze_default_66 = unsqueeze_default_67 = unsqueeze_default_68 = None
    getitem_146: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_22[0];  _scaled_dot_product_efficient_attention_default_22 = None
    squeeze_dim_22: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_146, 0);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_33: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_22, [1, 16, 128, 64]);  squeeze_dim_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_34: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_13, [1, 128, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_35: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_34, [128, 1024]);  view_34 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    
    # No stacktrace found for following nodes
    mm_default_182: "f32[128, 1024]" = torch.ops.aten.mm.default(view_35, permute_19);  view_35 = permute_19 = None
    add_tensor_182: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_182, arg28_1);  mm_default_182 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_36: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_182, [1, 128, 1024]);  add_tensor_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_7, view_36);  add_7 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  getitem_7 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_12: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
    add_12: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_37: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_12, [128, 1024]);  add_12 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    
    # No stacktrace found for following nodes
    mm_default_181: "f32[128, 4096]" = torch.ops.aten.mm.default(view_37, permute_20);  view_37 = permute_20 = None
    add_tensor_181: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_181, arg32_1);  mm_default_181 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_38: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_181, [1, 128, 4096]);  add_tensor_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_15: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_13: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_14, add_13);  mul_14 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_39: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_16, [128, 4096]);  mul_16 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    
    # No stacktrace found for following nodes
    mm_default_180: "f32[128, 1024]" = torch.ops.aten.mm.default(view_39, permute_21);  view_39 = permute_21 = None
    add_tensor_180: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_180, arg34_1);  mm_default_180 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_40: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_180, [1, 128, 1024]);  add_tensor_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_10, view_40);  add_10 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  getitem_9 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    mul_17: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_41: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_16, [128, 1024])
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    
    # No stacktrace found for following nodes
    mm_default_179: "f32[128, 1024]" = torch.ops.aten.mm.default(view_41, permute_22);  view_41 = permute_22 = None
    add_tensor_179: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_179, arg38_1);  mm_default_179 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_42: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_179, [1, 128, 1024]);  add_tensor_179 = None
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_42, 0.125);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_19, [1, 128, 16, 64]);  mul_19 = None
    permute_27: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_19: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_50: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_19, [16, -1, 64]);  clone_19 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_63: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_43: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_16, [128, 1024])
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    
    # No stacktrace found for following nodes
    mm_default_178: "f32[128, 1024]" = torch.ops.aten.mm.default(view_43, permute_23);  view_43 = permute_23 = None
    add_tensor_178: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_178, arg40_1);  mm_default_178 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_44: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_178, [1, 128, 1024]);  add_tensor_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_44, [1, -1, 16, 64]);  view_44 = None
    permute_24: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_17: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_51: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_17, [16, -1, 64]);  clone_17 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_64: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_51, 0);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_46: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_16, [128, 1024]);  add_16 = None
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    
    # No stacktrace found for following nodes
    mm_default_177: "f32[128, 1024]" = torch.ops.aten.mm.default(view_46, permute_25);  view_46 = permute_25 = None
    add_tensor_177: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_177, arg42_1);  mm_default_177 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_47: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_177, [1, 128, 1024]);  add_tensor_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_47, [1, -1, 16, 64]);  view_47 = None
    permute_26: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_18: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_52: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_18, [16, -1, 64]);  clone_18 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_65: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_52, 0);  view_52 = None
    _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_63, unsqueeze_default_64, unsqueeze_default_65, None, True, scale = 1.0);  unsqueeze_default_63 = unsqueeze_default_64 = unsqueeze_default_65 = None
    getitem_145: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_21[0];  _scaled_dot_product_efficient_attention_default_21 = None
    squeeze_dim_21: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_145, 0);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_53: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_21, [1, 16, 128, 64]);  squeeze_dim_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_54: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_21, [1, 128, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_55: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_54, [128, 1024]);  view_54 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    
    # No stacktrace found for following nodes
    mm_default_176: "f32[128, 1024]" = torch.ops.aten.mm.default(view_55, permute_30);  view_55 = permute_30 = None
    add_tensor_176: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_176, arg44_1);  mm_default_176 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_56: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_176, [1, 128, 1024]);  add_tensor_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_17: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_14, view_56);  add_14 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  getitem_11 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_21: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
    add_19: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_57: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_19, [128, 1024]);  add_19 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    
    # No stacktrace found for following nodes
    mm_default_175: "f32[128, 4096]" = torch.ops.aten.mm.default(view_57, permute_31);  view_57 = permute_31 = None
    add_tensor_175: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_175, arg48_1);  mm_default_175 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_58: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_175, [1, 128, 4096]);  add_tensor_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_23: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_2: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_20: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_22, add_20);  mul_22 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_59: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_24, [128, 4096]);  mul_24 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    
    # No stacktrace found for following nodes
    mm_default_174: "f32[128, 1024]" = torch.ops.aten.mm.default(view_59, permute_32);  view_59 = permute_32 = None
    add_tensor_174: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_174, arg50_1);  mm_default_174 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_60: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_174, [1, 128, 1024]);  add_tensor_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_21: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_17, view_60);  add_17 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_13);  getitem_13 = None
    add_22: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_26: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
    add_23: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_61: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_23, [128, 1024])
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    
    # No stacktrace found for following nodes
    mm_default_173: "f32[128, 1024]" = torch.ops.aten.mm.default(view_61, permute_33);  view_61 = permute_33 = None
    add_tensor_173: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_173, arg54_1);  mm_default_173 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_62: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_173, [1, 128, 1024]);  add_tensor_173 = None
    mul_27: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_62, 0.125);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_27, [1, 128, 16, 64]);  mul_27 = None
    permute_38: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_27: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_70: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_27, [16, -1, 64]);  clone_27 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_60: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_70, 0);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_63: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_23, [128, 1024])
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    
    # No stacktrace found for following nodes
    mm_default_172: "f32[128, 1024]" = torch.ops.aten.mm.default(view_63, permute_34);  view_63 = permute_34 = None
    add_tensor_172: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_172, arg56_1);  mm_default_172 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_64: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_172, [1, 128, 1024]);  add_tensor_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_64, [1, -1, 16, 64]);  view_64 = None
    permute_35: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_25: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_71: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_25, [16, -1, 64]);  clone_25 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_61: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_66: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_23, [128, 1024]);  add_23 = None
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    
    # No stacktrace found for following nodes
    mm_default_171: "f32[128, 1024]" = torch.ops.aten.mm.default(view_66, permute_36);  view_66 = permute_36 = None
    add_tensor_171: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_171, arg58_1);  mm_default_171 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_67: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_171, [1, 128, 1024]);  add_tensor_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_67, [1, -1, 16, 64]);  view_67 = None
    permute_37: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_26: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_72: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_26, [16, -1, 64]);  clone_26 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_62: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
    _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_60, unsqueeze_default_61, unsqueeze_default_62, None, True, scale = 1.0);  unsqueeze_default_60 = unsqueeze_default_61 = unsqueeze_default_62 = None
    getitem_144: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_20[0];  _scaled_dot_product_efficient_attention_default_20 = None
    squeeze_dim_20: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_144, 0);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_73: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_20, [1, 16, 128, 64]);  squeeze_dim_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_74: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_29, [1, 128, 1024]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_75: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_74, [128, 1024]);  view_74 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    
    # No stacktrace found for following nodes
    mm_default_170: "f32[128, 1024]" = torch.ops.aten.mm.default(view_75, permute_41);  view_75 = permute_41 = None
    add_tensor_170: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_170, arg60_1);  mm_default_170 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_76: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_170, [1, 128, 1024]);  add_tensor_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_24: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_21, view_76);  add_21 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  getitem_15 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_29: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
    add_26: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_77: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_26, [128, 1024]);  add_26 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    
    # No stacktrace found for following nodes
    mm_default_169: "f32[128, 4096]" = torch.ops.aten.mm.default(view_77, permute_42);  view_77 = permute_42 = None
    add_tensor_169: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_169, arg64_1);  mm_default_169 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_78: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_169, [1, 128, 4096]);  add_tensor_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_3: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_27: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, add_27);  mul_30 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_79: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_32, [128, 4096]);  mul_32 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    
    # No stacktrace found for following nodes
    mm_default_168: "f32[128, 1024]" = torch.ops.aten.mm.default(view_79, permute_43);  view_79 = permute_43 = None
    add_tensor_168: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_168, arg66_1);  mm_default_168 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_80: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_168, [1, 128, 1024]);  add_tensor_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_28: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_24, view_80);  add_24 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_28, getitem_17);  getitem_17 = None
    add_29: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    mul_33: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
    add_30: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_81: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_30, [128, 1024])
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    
    # No stacktrace found for following nodes
    mm_default_167: "f32[128, 1024]" = torch.ops.aten.mm.default(view_81, permute_44);  view_81 = permute_44 = None
    add_tensor_167: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_167, arg70_1);  mm_default_167 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_82: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_167, [1, 128, 1024]);  add_tensor_167 = None
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_82, 0.125);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_35, [1, 128, 16, 64]);  mul_35 = None
    permute_49: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_35: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_90: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_35, [16, -1, 64]);  clone_35 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_57: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_90, 0);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_83: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_30, [128, 1024])
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    
    # No stacktrace found for following nodes
    mm_default_166: "f32[128, 1024]" = torch.ops.aten.mm.default(view_83, permute_45);  view_83 = permute_45 = None
    add_tensor_166: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_166, arg72_1);  mm_default_166 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_84: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_166, [1, 128, 1024]);  add_tensor_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_84, [1, -1, 16, 64]);  view_84 = None
    permute_46: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    clone_33: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_91: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_33, [16, -1, 64]);  clone_33 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_58: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_91, 0);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_86: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_30, [128, 1024]);  add_30 = None
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    
    # No stacktrace found for following nodes
    mm_default_165: "f32[128, 1024]" = torch.ops.aten.mm.default(view_86, permute_47);  view_86 = permute_47 = None
    add_tensor_165: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_165, arg74_1);  mm_default_165 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_87: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_165, [1, 128, 1024]);  add_tensor_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_87, [1, -1, 16, 64]);  view_87 = None
    permute_48: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
    clone_34: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_92: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_34, [16, -1, 64]);  clone_34 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_59: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_92, 0);  view_92 = None
    _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_57, unsqueeze_default_58, unsqueeze_default_59, None, True, scale = 1.0);  unsqueeze_default_57 = unsqueeze_default_58 = unsqueeze_default_59 = None
    getitem_143: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_19[0];  _scaled_dot_product_efficient_attention_default_19 = None
    squeeze_dim_19: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_143, 0);  getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_93: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_19, [1, 16, 128, 64]);  squeeze_dim_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_94: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_37, [1, 128, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_95: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_94, [128, 1024]);  view_94 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    
    # No stacktrace found for following nodes
    mm_default_164: "f32[128, 1024]" = torch.ops.aten.mm.default(view_95, permute_52);  view_95 = permute_52 = None
    add_tensor_164: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_164, arg76_1);  mm_default_164 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_96: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_164, [1, 128, 1024]);  add_tensor_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_31: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_28, view_96);  add_28 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  getitem_19 = None
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
    add_33: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_97: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_33, [128, 1024]);  add_33 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    
    # No stacktrace found for following nodes
    mm_default_163: "f32[128, 4096]" = torch.ops.aten.mm.default(view_97, permute_53);  view_97 = permute_53 = None
    add_tensor_163: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_163, arg80_1);  mm_default_163 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_98: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_163, [1, 128, 4096]);  add_tensor_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_39: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_4: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_34: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_38, add_34);  mul_38 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_99: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_40, [128, 4096]);  mul_40 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_162: "f32[128, 1024]" = torch.ops.aten.mm.default(view_99, permute_54);  view_99 = permute_54 = None
    add_tensor_162: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_162, arg82_1);  mm_default_162 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_100: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_162, [1, 128, 1024]);  add_tensor_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_35: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_31, view_100);  add_31 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_15: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_35, getitem_21);  getitem_21 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_41: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_42: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
    add_37: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_101: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_37, [128, 1024])
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    
    # No stacktrace found for following nodes
    mm_default_161: "f32[128, 1024]" = torch.ops.aten.mm.default(view_101, permute_55);  view_101 = permute_55 = None
    add_tensor_161: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_161, arg86_1);  mm_default_161 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_102: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_161, [1, 128, 1024]);  add_tensor_161 = None
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_102, 0.125);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_43, [1, 128, 16, 64]);  mul_43 = None
    permute_60: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_43: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_110: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_43, [16, -1, 64]);  clone_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_54: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_103: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_37, [128, 1024])
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    
    # No stacktrace found for following nodes
    mm_default_160: "f32[128, 1024]" = torch.ops.aten.mm.default(view_103, permute_56);  view_103 = permute_56 = None
    add_tensor_160: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_160, arg88_1);  mm_default_160 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_104: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_160, [1, 128, 1024]);  add_tensor_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_104, [1, -1, 16, 64]);  view_104 = None
    permute_57: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    clone_41: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_111: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_41, [16, -1, 64]);  clone_41 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_55: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_111, 0);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_106: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_37, [128, 1024]);  add_37 = None
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    
    # No stacktrace found for following nodes
    mm_default_159: "f32[128, 1024]" = torch.ops.aten.mm.default(view_106, permute_58);  view_106 = permute_58 = None
    add_tensor_159: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_159, arg90_1);  mm_default_159 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_107: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_159, [1, 128, 1024]);  add_tensor_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_108: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_107, [1, -1, 16, 64]);  view_107 = None
    permute_59: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    clone_42: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_112: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_42, [16, -1, 64]);  clone_42 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_56: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_112, 0);  view_112 = None
    _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_54, unsqueeze_default_55, unsqueeze_default_56, None, True, scale = 1.0);  unsqueeze_default_54 = unsqueeze_default_55 = unsqueeze_default_56 = None
    getitem_142: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_18[0];  _scaled_dot_product_efficient_attention_default_18 = None
    squeeze_dim_18: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_142, 0);  getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_113: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_18, [1, 16, 128, 64]);  squeeze_dim_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_114: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_45, [1, 128, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_115: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_114, [128, 1024]);  view_114 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    
    # No stacktrace found for following nodes
    mm_default_158: "f32[128, 1024]" = torch.ops.aten.mm.default(view_115, permute_63);  view_115 = permute_63 = None
    add_tensor_158: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_158, arg92_1);  mm_default_158 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_116: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_158, [1, 128, 1024]);  add_tensor_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_35, view_116);  add_35 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  getitem_23 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_44: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_45: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
    add_40: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_117: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_40, [128, 1024]);  add_40 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    
    # No stacktrace found for following nodes
    mm_default_157: "f32[128, 4096]" = torch.ops.aten.mm.default(view_117, permute_64);  view_117 = permute_64 = None
    add_tensor_157: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_157, arg96_1);  mm_default_157 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_118: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_157, [1, 128, 4096]);  add_tensor_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_47: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_5: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_41: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_46, add_41);  mul_46 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_119: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_48, [128, 4096]);  mul_48 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    
    # No stacktrace found for following nodes
    mm_default_156: "f32[128, 1024]" = torch.ops.aten.mm.default(view_119, permute_65);  view_119 = permute_65 = None
    add_tensor_156: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_156, arg98_1);  mm_default_156 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_120: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_156, [1, 128, 1024]);  add_tensor_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_42: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_38, view_120);  add_38 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_18: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_42, getitem_25);  getitem_25 = None
    add_43: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    mul_49: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_50: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
    add_44: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_121: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_44, [128, 1024])
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    
    # No stacktrace found for following nodes
    mm_default_155: "f32[128, 1024]" = torch.ops.aten.mm.default(view_121, permute_66);  view_121 = permute_66 = None
    add_tensor_155: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_155, arg102_1);  mm_default_155 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_122: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_155, [1, 128, 1024]);  add_tensor_155 = None
    mul_51: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_122, 0.125);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_129: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_51, [1, 128, 16, 64]);  mul_51 = None
    permute_71: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    clone_51: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_130: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_51, [16, -1, 64]);  clone_51 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_51: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_130, 0);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_123: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_44, [128, 1024])
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    
    # No stacktrace found for following nodes
    mm_default_154: "f32[128, 1024]" = torch.ops.aten.mm.default(view_123, permute_67);  view_123 = permute_67 = None
    add_tensor_154: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_154, arg104_1);  mm_default_154 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_124: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_154, [1, 128, 1024]);  add_tensor_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_125: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_124, [1, -1, 16, 64]);  view_124 = None
    permute_68: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
    clone_49: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_131: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_49, [16, -1, 64]);  clone_49 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_52: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_131, 0);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_126: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_44, [128, 1024]);  add_44 = None
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    
    # No stacktrace found for following nodes
    mm_default_153: "f32[128, 1024]" = torch.ops.aten.mm.default(view_126, permute_69);  view_126 = permute_69 = None
    add_tensor_153: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_153, arg106_1);  mm_default_153 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_127: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_153, [1, 128, 1024]);  add_tensor_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_128: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_127, [1, -1, 16, 64]);  view_127 = None
    permute_70: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    clone_50: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_132: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_50, [16, -1, 64]);  clone_50 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_53: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_132, 0);  view_132 = None
    _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_51, unsqueeze_default_52, unsqueeze_default_53, None, True, scale = 1.0);  unsqueeze_default_51 = unsqueeze_default_52 = unsqueeze_default_53 = None
    getitem_141: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_17[0];  _scaled_dot_product_efficient_attention_default_17 = None
    squeeze_dim_17: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_141, 0);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_133: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_17, [1, 16, 128, 64]);  squeeze_dim_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_134: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_53, [1, 128, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_135: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_134, [128, 1024]);  view_134 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    
    # No stacktrace found for following nodes
    mm_default_152: "f32[128, 1024]" = torch.ops.aten.mm.default(view_135, permute_74);  view_135 = permute_74 = None
    add_tensor_152: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_152, arg108_1);  mm_default_152 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_136: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_152, [1, 128, 1024]);  add_tensor_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_45: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_42, view_136);  add_42 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_20: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_27);  getitem_27 = None
    add_46: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_52: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_53: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
    add_47: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_137: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_47, [128, 1024]);  add_47 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    
    # No stacktrace found for following nodes
    mm_default_151: "f32[128, 4096]" = torch.ops.aten.mm.default(view_137, permute_75);  view_137 = permute_75 = None
    add_tensor_151: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_151, arg112_1);  mm_default_151 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_138: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_151, [1, 128, 4096]);  add_tensor_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
    mul_55: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
    erf_6: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_48: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_48);  mul_54 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_139: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_56, [128, 4096]);  mul_56 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    
    # No stacktrace found for following nodes
    mm_default_150: "f32[128, 1024]" = torch.ops.aten.mm.default(view_139, permute_76);  view_139 = permute_76 = None
    add_tensor_150: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_150, arg114_1);  mm_default_150 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_140: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_150, [1, 128, 1024]);  add_tensor_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_49: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_45, view_140);  add_45 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_21: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_49, getitem_29);  getitem_29 = None
    add_50: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_57: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_58: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
    add_51: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_141: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_51, [128, 1024])
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    
    # No stacktrace found for following nodes
    mm_default_149: "f32[128, 1024]" = torch.ops.aten.mm.default(view_141, permute_77);  view_141 = permute_77 = None
    add_tensor_149: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_149, arg118_1);  mm_default_149 = arg118_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_142: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_149, [1, 128, 1024]);  add_tensor_149 = None
    mul_59: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_59, [1, 128, 16, 64]);  mul_59 = None
    permute_82: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_59: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_150: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_59, [16, -1, 64]);  clone_59 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_48: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_150, 0);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_143: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_51, [128, 1024])
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    
    # No stacktrace found for following nodes
    mm_default_148: "f32[128, 1024]" = torch.ops.aten.mm.default(view_143, permute_78);  view_143 = permute_78 = None
    add_tensor_148: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_148, arg120_1);  mm_default_148 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_144: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_148, [1, 128, 1024]);  add_tensor_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_144, [1, -1, 16, 64]);  view_144 = None
    permute_79: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_57: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_151: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_57, [16, -1, 64]);  clone_57 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_49: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_151, 0);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_146: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_51, [128, 1024]);  add_51 = None
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    
    # No stacktrace found for following nodes
    mm_default_147: "f32[128, 1024]" = torch.ops.aten.mm.default(view_146, permute_80);  view_146 = permute_80 = None
    add_tensor_147: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_147, arg122_1);  mm_default_147 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_147: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_147, [1, 128, 1024]);  add_tensor_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_147, [1, -1, 16, 64]);  view_147 = None
    permute_81: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_58: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_152: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_58, [16, -1, 64]);  clone_58 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_50: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_152, 0);  view_152 = None
    _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_48, unsqueeze_default_49, unsqueeze_default_50, None, True, scale = 1.0);  unsqueeze_default_48 = unsqueeze_default_49 = unsqueeze_default_50 = None
    getitem_140: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_16[0];  _scaled_dot_product_efficient_attention_default_16 = None
    squeeze_dim_16: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_140, 0);  getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_153: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_16, [1, 16, 128, 64]);  squeeze_dim_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_154: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_61, [1, 128, 1024]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_155: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_154, [128, 1024]);  view_154 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    
    # No stacktrace found for following nodes
    mm_default_146: "f32[128, 1024]" = torch.ops.aten.mm.default(view_155, permute_85);  view_155 = permute_85 = None
    add_tensor_146: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_146, arg124_1);  mm_default_146 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_156: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_146, [1, 128, 1024]);  add_tensor_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_52: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_49, view_156);  add_49 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_23: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_52, getitem_31);  getitem_31 = None
    add_53: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_60: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_61: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
    add_54: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_157: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_54, [128, 1024]);  add_54 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    
    # No stacktrace found for following nodes
    mm_default_145: "f32[128, 4096]" = torch.ops.aten.mm.default(view_157, permute_86);  view_157 = permute_86 = None
    add_tensor_145: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_145, arg128_1);  mm_default_145 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_158: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_145, [1, 128, 4096]);  add_tensor_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_63: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_7: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_55: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_62, add_55);  mul_62 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_159: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_64, [128, 4096]);  mul_64 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    
    # No stacktrace found for following nodes
    mm_default_144: "f32[128, 1024]" = torch.ops.aten.mm.default(view_159, permute_87);  view_159 = permute_87 = None
    add_tensor_144: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_144, arg130_1);  mm_default_144 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_160: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_144, [1, 128, 1024]);  add_tensor_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_56: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_52, view_160);  add_52 = view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_24: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_56, getitem_33);  getitem_33 = None
    add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    mul_65: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_66: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
    add_58: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_161: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_58, [128, 1024])
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    
    # No stacktrace found for following nodes
    mm_default_143: "f32[128, 1024]" = torch.ops.aten.mm.default(view_161, permute_88);  view_161 = permute_88 = None
    add_tensor_143: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_143, arg134_1);  mm_default_143 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_162: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_143, [1, 128, 1024]);  add_tensor_143 = None
    mul_67: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_162, 0.125);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_67, [1, 128, 16, 64]);  mul_67 = None
    permute_93: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
    clone_67: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_170: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_67, [16, -1, 64]);  clone_67 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_45: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_170, 0);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_163: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_58, [128, 1024])
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    
    # No stacktrace found for following nodes
    mm_default_142: "f32[128, 1024]" = torch.ops.aten.mm.default(view_163, permute_89);  view_163 = permute_89 = None
    add_tensor_142: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_142, arg136_1);  mm_default_142 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_164: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_142, [1, 128, 1024]);  add_tensor_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_164, [1, -1, 16, 64]);  view_164 = None
    permute_90: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    clone_65: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_171: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_65, [16, -1, 64]);  clone_65 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_46: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_171, 0);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_166: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_58, [128, 1024]);  add_58 = None
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    
    # No stacktrace found for following nodes
    mm_default_141: "f32[128, 1024]" = torch.ops.aten.mm.default(view_166, permute_91);  view_166 = permute_91 = None
    add_tensor_141: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_141, arg138_1);  mm_default_141 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_167: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_141, [1, 128, 1024]);  add_tensor_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_168: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_167, [1, -1, 16, 64]);  view_167 = None
    permute_92: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_66: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_172: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_66, [16, -1, 64]);  clone_66 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_47: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_172, 0);  view_172 = None
    _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_45, unsqueeze_default_46, unsqueeze_default_47, None, True, scale = 1.0);  unsqueeze_default_45 = unsqueeze_default_46 = unsqueeze_default_47 = None
    getitem_139: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
    squeeze_dim_15: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_139, 0);  getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_173: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_15, [1, 16, 128, 64]);  squeeze_dim_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_69: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_174: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_69, [1, 128, 1024]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_175: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_174, [128, 1024]);  view_174 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    
    # No stacktrace found for following nodes
    mm_default_140: "f32[128, 1024]" = torch.ops.aten.mm.default(view_175, permute_96);  view_175 = permute_96 = None
    add_tensor_140: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_140, arg140_1);  mm_default_140 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_176: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_140, [1, 128, 1024]);  add_tensor_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_59: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_56, view_176);  add_56 = view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_26: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_59, getitem_35);  getitem_35 = None
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_68: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_69: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_68, arg141_1);  mul_68 = arg141_1 = None
    add_61: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_177: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_61, [128, 1024]);  add_61 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    
    # No stacktrace found for following nodes
    mm_default_139: "f32[128, 4096]" = torch.ops.aten.mm.default(view_177, permute_97);  view_177 = permute_97 = None
    add_tensor_139: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_139, arg144_1);  mm_default_139 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_178: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_139, [1, 128, 4096]);  add_tensor_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_70: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
    erf_8: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_62: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_72: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, add_62);  mul_70 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_179: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_72, [128, 4096]);  mul_72 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    
    # No stacktrace found for following nodes
    mm_default_138: "f32[128, 1024]" = torch.ops.aten.mm.default(view_179, permute_98);  view_179 = permute_98 = None
    add_tensor_138: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_138, arg146_1);  mm_default_138 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_180: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_138, [1, 128, 1024]);  add_tensor_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_63: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_59, view_180);  add_59 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_27: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_63, getitem_37);  getitem_37 = None
    add_64: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    mul_73: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_74: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg147_1);  mul_73 = arg147_1 = None
    add_65: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_181: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_65, [128, 1024])
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    
    # No stacktrace found for following nodes
    mm_default_137: "f32[128, 1024]" = torch.ops.aten.mm.default(view_181, permute_99);  view_181 = permute_99 = None
    add_tensor_137: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_137, arg150_1);  mm_default_137 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_182: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_137, [1, 128, 1024]);  add_tensor_137 = None
    mul_75: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_182, 0.125);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_189: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_75, [1, 128, 16, 64]);  mul_75 = None
    permute_104: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
    clone_75: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_190: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_75, [16, -1, 64]);  clone_75 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_42: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_190, 0);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_183: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_65, [128, 1024])
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    
    # No stacktrace found for following nodes
    mm_default_136: "f32[128, 1024]" = torch.ops.aten.mm.default(view_183, permute_100);  view_183 = permute_100 = None
    add_tensor_136: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_136, arg152_1);  mm_default_136 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_184: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_136, [1, 128, 1024]);  add_tensor_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_185: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_184, [1, -1, 16, 64]);  view_184 = None
    permute_101: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    clone_73: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_191: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_73, [16, -1, 64]);  clone_73 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_43: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_191, 0);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_186: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_65, [128, 1024]);  add_65 = None
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    
    # No stacktrace found for following nodes
    mm_default_135: "f32[128, 1024]" = torch.ops.aten.mm.default(view_186, permute_102);  view_186 = permute_102 = None
    add_tensor_135: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_135, arg154_1);  mm_default_135 = arg154_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_187: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_135, [1, 128, 1024]);  add_tensor_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_188: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_187, [1, -1, 16, 64]);  view_187 = None
    permute_103: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    clone_74: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_192: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_74, [16, -1, 64]);  clone_74 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_44: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_192, 0);  view_192 = None
    _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_42, unsqueeze_default_43, unsqueeze_default_44, None, True, scale = 1.0);  unsqueeze_default_42 = unsqueeze_default_43 = unsqueeze_default_44 = None
    getitem_138: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
    squeeze_dim_14: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_138, 0);  getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_193: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_14, [1, 16, 128, 64]);  squeeze_dim_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_77: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_194: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_77, [1, 128, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_195: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_194, [128, 1024]);  view_194 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    
    # No stacktrace found for following nodes
    mm_default_134: "f32[128, 1024]" = torch.ops.aten.mm.default(view_195, permute_107);  view_195 = permute_107 = None
    add_tensor_134: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_134, arg156_1);  mm_default_134 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_196: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_134, [1, 128, 1024]);  add_tensor_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_66: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_63, view_196);  add_63 = view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_29: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_66, getitem_39);  getitem_39 = None
    add_67: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_76: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_77: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_76, arg157_1);  mul_76 = arg157_1 = None
    add_68: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_197: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_68, [128, 1024]);  add_68 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    
    # No stacktrace found for following nodes
    mm_default_133: "f32[128, 4096]" = torch.ops.aten.mm.default(view_197, permute_108);  view_197 = permute_108 = None
    add_tensor_133: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_133, arg160_1);  mm_default_133 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_198: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_133, [1, 128, 4096]);  add_tensor_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_78: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_198, 0.5)
    mul_79: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476);  view_198 = None
    erf_9: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_69: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_80: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_78, add_69);  mul_78 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_199: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_80, [128, 4096]);  mul_80 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    
    # No stacktrace found for following nodes
    mm_default_132: "f32[128, 1024]" = torch.ops.aten.mm.default(view_199, permute_109);  view_199 = permute_109 = None
    add_tensor_132: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_132, arg162_1);  mm_default_132 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_200: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_132, [1, 128, 1024]);  add_tensor_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_70: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_66, view_200);  add_66 = view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_30: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_70, getitem_41);  getitem_41 = None
    add_71: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_81: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_82: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_81, arg163_1);  mul_81 = arg163_1 = None
    add_72: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_201: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_72, [128, 1024])
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    
    # No stacktrace found for following nodes
    mm_default_131: "f32[128, 1024]" = torch.ops.aten.mm.default(view_201, permute_110);  view_201 = permute_110 = None
    add_tensor_131: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_131, arg166_1);  mm_default_131 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_202: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_131, [1, 128, 1024]);  add_tensor_131 = None
    mul_83: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_202, 0.125);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_209: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_83, [1, 128, 16, 64]);  mul_83 = None
    permute_115: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_83: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_210: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_83, [16, -1, 64]);  clone_83 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_39: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_210, 0);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_203: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_72, [128, 1024])
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    
    # No stacktrace found for following nodes
    mm_default_130: "f32[128, 1024]" = torch.ops.aten.mm.default(view_203, permute_111);  view_203 = permute_111 = None
    add_tensor_130: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_130, arg168_1);  mm_default_130 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_204: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_130, [1, 128, 1024]);  add_tensor_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_205: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_204, [1, -1, 16, 64]);  view_204 = None
    permute_112: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    clone_81: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_211: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_81, [16, -1, 64]);  clone_81 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_40: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_211, 0);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_206: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_72, [128, 1024]);  add_72 = None
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    
    # No stacktrace found for following nodes
    mm_default_129: "f32[128, 1024]" = torch.ops.aten.mm.default(view_206, permute_113);  view_206 = permute_113 = None
    add_tensor_129: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_129, arg170_1);  mm_default_129 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_207: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_129, [1, 128, 1024]);  add_tensor_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_208: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_207, [1, -1, 16, 64]);  view_207 = None
    permute_114: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_82: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_212: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_82, [16, -1, 64]);  clone_82 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_41: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_212, 0);  view_212 = None
    _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_39, unsqueeze_default_40, unsqueeze_default_41, None, True, scale = 1.0);  unsqueeze_default_39 = unsqueeze_default_40 = unsqueeze_default_41 = None
    getitem_137: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
    squeeze_dim_13: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_137, 0);  getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_213: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_13, [1, 16, 128, 64]);  squeeze_dim_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_85: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_214: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_85, [1, 128, 1024]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_215: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_214, [128, 1024]);  view_214 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    
    # No stacktrace found for following nodes
    mm_default_128: "f32[128, 1024]" = torch.ops.aten.mm.default(view_215, permute_118);  view_215 = permute_118 = None
    add_tensor_128: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_128, arg172_1);  mm_default_128 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_128, [1, 128, 1024]);  add_tensor_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_73: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_70, view_216);  add_70 = view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_32: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_43);  getitem_43 = None
    add_74: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_84: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_85: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_84, arg173_1);  mul_84 = arg173_1 = None
    add_75: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_217: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_75, [128, 1024]);  add_75 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    
    # No stacktrace found for following nodes
    mm_default_127: "f32[128, 4096]" = torch.ops.aten.mm.default(view_217, permute_119);  view_217 = permute_119 = None
    add_tensor_127: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_127, arg176_1);  mm_default_127 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_218: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_127, [1, 128, 4096]);  add_tensor_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_86: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_218, 0.5)
    mul_87: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_218, 0.7071067811865476);  view_218 = None
    erf_10: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_76: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_88: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_86, add_76);  mul_86 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_219: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_88, [128, 4096]);  mul_88 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    
    # No stacktrace found for following nodes
    mm_default_126: "f32[128, 1024]" = torch.ops.aten.mm.default(view_219, permute_120);  view_219 = permute_120 = None
    add_tensor_126: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_126, arg178_1);  mm_default_126 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_220: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_126, [1, 128, 1024]);  add_tensor_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_77: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_73, view_220);  add_73 = view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:335, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_33: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_77, getitem_45);  getitem_45 = None
    add_78: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_89: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_90: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_89, arg179_1);  mul_89 = arg179_1 = None
    add_79: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_221: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_79, [128, 1024])
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    
    # No stacktrace found for following nodes
    mm_default_125: "f32[128, 1024]" = torch.ops.aten.mm.default(view_221, permute_121);  view_221 = permute_121 = None
    add_tensor_125: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_125, arg182_1);  mm_default_125 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_222: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_125, [1, 128, 1024]);  add_tensor_125 = None
    mul_91: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_222, 0.125);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_229: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_91, [1, 128, 16, 64]);  mul_91 = None
    permute_126: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    clone_91: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_230: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_91, [16, -1, 64]);  clone_91 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_36: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_230, 0);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_223: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_79, [128, 1024])
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    
    # No stacktrace found for following nodes
    mm_default_124: "f32[128, 1024]" = torch.ops.aten.mm.default(view_223, permute_122);  view_223 = permute_122 = None
    add_tensor_124: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_124, arg184_1);  mm_default_124 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_224: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_124, [1, 128, 1024]);  add_tensor_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_225: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_224, [1, -1, 16, 64]);  view_224 = None
    permute_123: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    clone_89: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_231: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_89, [16, -1, 64]);  clone_89 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_37: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_231, 0);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_226: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_79, [128, 1024]);  add_79 = None
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    
    # No stacktrace found for following nodes
    mm_default_123: "f32[128, 1024]" = torch.ops.aten.mm.default(view_226, permute_124);  view_226 = permute_124 = None
    add_tensor_123: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_123, arg186_1);  mm_default_123 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_227: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_123, [1, 128, 1024]);  add_tensor_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_228: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_227, [1, -1, 16, 64]);  view_227 = None
    permute_125: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    clone_90: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_232: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_90, [16, -1, 64]);  clone_90 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_38: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_232, 0);  view_232 = None
    _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_36, unsqueeze_default_37, unsqueeze_default_38, None, True, scale = 1.0);  unsqueeze_default_36 = unsqueeze_default_37 = unsqueeze_default_38 = None
    getitem_136: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
    squeeze_dim_12: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_136, 0);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_233: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_12, [1, 16, 128, 64]);  squeeze_dim_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_93: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_234: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_93, [1, 128, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_235: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_234, [128, 1024]);  view_234 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    
    # No stacktrace found for following nodes
    mm_default_122: "f32[128, 1024]" = torch.ops.aten.mm.default(view_235, permute_129);  view_235 = permute_129 = None
    add_tensor_122: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_122, arg188_1);  mm_default_122 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_236: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_122, [1, 128, 1024]);  add_tensor_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:343, code: hidden_states = residual + hidden_states
    add_80: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_77, view_236);  add_77 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:346, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_35: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_80, getitem_47);  getitem_47 = None
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_92: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_93: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_92, arg189_1);  mul_92 = arg189_1 = None
    add_82: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_237: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_82, [128, 1024]);  add_82 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    
    # No stacktrace found for following nodes
    mm_default_121: "f32[128, 4096]" = torch.ops.aten.mm.default(view_237, permute_130);  view_237 = permute_130 = None
    add_tensor_121: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_121, arg192_1);  mm_default_121 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:347, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_238: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_121, [1, 128, 4096]);  add_tensor_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_238, 0.5)
    mul_95: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_238, 0.7071067811865476);  view_238 = None
    erf_11: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_83: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_96: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_94, add_83);  mul_94 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_239: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_96, [128, 4096]);  mul_96 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    
    # No stacktrace found for following nodes
    mm_default_120: "f32[128, 1024]" = torch.ops.aten.mm.default(view_239, permute_131);  view_239 = permute_131 = None
    add_tensor_120: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_120, arg194_1);  mm_default_120 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:349, code: hidden_states = self.fc2(hidden_states)
    view_240: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_120, [1, 128, 1024]);  add_tensor_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:351, code: hidden_states = residual + hidden_states
    add_84: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_80, view_240);  add_80 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:832, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1029, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_241: "i64[1, 128]" = torch.ops.aten.reshape.default(arg514_1, [-1, 128]);  arg514_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1039, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding_2: "f32[1, 128, 1024]" = torch.ops.aten.embedding.default(arg2_1, view_241, 0);  arg2_1 = view_241 = None
    mul_99: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:137, code: positions = torch.arange(
    iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[128, 1024]" = torch.ops.aten.embedding.default(arg1_1, iota_2);  arg1_1 = iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1053, code: hidden_states = inputs_embeds + positions
    add_88: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_99, embedding_3);  mul_99 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_37: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_88, getitem_51);  getitem_51 = None
    add_89: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_100: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
    mul_101: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_100, arg197_1);  mul_100 = arg197_1 = None
    add_90: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_101, arg198_1);  mul_101 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_243: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_90, [128, 1024])
    permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    
    # No stacktrace found for following nodes
    mm_default_119: "f32[128, 1024]" = torch.ops.aten.mm.default(view_243, permute_132);  view_243 = permute_132 = None
    add_tensor_119: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_119, arg200_1);  mm_default_119 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_244: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_119, [1, 128, 1024]);  add_tensor_119 = None
    mul_102: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_244, 0.125);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_251: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_102, [1, 128, 16, 64]);  mul_102 = None
    permute_137: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    clone_100: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_252: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_100, [16, -1, 64]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_245: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_90, [128, 1024])
    permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    
    # No stacktrace found for following nodes
    mm_default_118: "f32[128, 1024]" = torch.ops.aten.mm.default(view_245, permute_133);  view_245 = permute_133 = None
    add_tensor_118: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_118, arg202_1);  mm_default_118 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_246: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_118, [1, 128, 1024]);  add_tensor_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_247: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_246, [1, -1, 16, 64]);  view_246 = None
    permute_134: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
    clone_98: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_253: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_98, [16, -1, 64]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_138: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_252, permute_138);  view_252 = permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_255: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:84, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:85, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_87: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_242: "i64[128, 1]" = torch.ops.aten.reshape.default(add_87, [128, 1]);  add_87 = None
    lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota_1, view_242);  iota_1 = view_242 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:83, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:85, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where: "f32[128, 128]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_1: "f32[1, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 1, 128, 128]);  unsqueeze_3 = None
    add_91: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_255, expand_1);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_256: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_91, [16, 128, 128]);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_256, [-1], True)
    sub_38: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_256, amax_12);  view_256 = amax_12 = None
    exp_12: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_248: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_90, [128, 1024]);  add_90 = None
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    
    # No stacktrace found for following nodes
    mm_default_117: "f32[128, 1024]" = torch.ops.aten.mm.default(view_248, permute_135);  view_248 = permute_135 = None
    add_tensor_117: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_117, arg204_1);  mm_default_117 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_249: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_117, [1, 128, 1024]);  add_tensor_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_250: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_249, [1, -1, 16, 64]);  view_249 = None
    permute_136: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    clone_99: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_254: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_99, [16, -1, 64]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_12, view_254);  div_12 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_257: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_139: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_102: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_258: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_102, [1, 128, 1024]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_259: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_258, [128, 1024]);  view_258 = None
    permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    
    # No stacktrace found for following nodes
    mm_default_116: "f32[128, 1024]" = torch.ops.aten.mm.default(view_259, permute_140);  view_259 = permute_140 = None
    add_tensor_116: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_116, arg206_1);  mm_default_116 = arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_260: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_116, [1, 128, 1024]);  add_tensor_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_92: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_88, view_260);  add_88 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    sub_39: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_92, getitem_53);  getitem_53 = None
    add_93: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_103: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
    mul_104: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_103, arg207_1);  mul_103 = arg207_1 = None
    add_94: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_104, arg208_1);  mul_104 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_261: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_94, [128, 1024]);  add_94 = None
    permute_141: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
    
    # No stacktrace found for following nodes
    mm_default_115: "f32[128, 1024]" = torch.ops.aten.mm.default(view_261, permute_141);  view_261 = permute_141 = None
    add_tensor_115: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_115, arg210_1);  mm_default_115 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_262: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_115, [1, 128, 1024]);  add_tensor_115 = None
    mul_105: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_262, 0.125);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_269: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_105, [1, 128, 16, 64]);  mul_105 = None
    permute_146: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_106: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_270: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_106, [16, -1, 64]);  clone_106 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_33: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_270, 0);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:832, code: hidden_states = self.layer_norm(hidden_states)
    sub_36: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_84, getitem_49);  add_84 = getitem_49 = None
    add_85: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_97: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_98: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, arg195_1);  mul_97 = arg195_1 = None
    add_86: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_263: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_142: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
    
    # No stacktrace found for following nodes
    mm_default_114: "f32[128, 1024]" = torch.ops.aten.mm.default(view_263, permute_142);  view_263 = permute_142 = None
    add_tensor_114: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_114, arg212_1);  mm_default_114 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_264: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_114, [1, 128, 1024]);  add_tensor_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_265: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_264, [1, -1, 16, 64]);  view_264 = None
    permute_143: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
    clone_104: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_271: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_104, [16, -1, 64]);  clone_104 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_34: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_271, 0);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_266: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    
    # No stacktrace found for following nodes
    mm_default_113: "f32[128, 1024]" = torch.ops.aten.mm.default(view_266, permute_144);  view_266 = permute_144 = None
    add_tensor_113: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_113, arg214_1);  mm_default_113 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_267: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_113, [1, 128, 1024]);  add_tensor_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_268: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_267, [1, -1, 16, 64]);  view_267 = None
    permute_145: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
    clone_105: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_272: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_105, [16, -1, 64]);  clone_105 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_35: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_272, 0);  view_272 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, None, True, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
    getitem_135: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
    squeeze_dim_11: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_135, 0);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_273: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_11, [1, 16, 128, 64]);  squeeze_dim_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_148: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_108: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_274: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_108, [1, 128, 1024]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_275: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_274, [128, 1024]);  view_274 = None
    permute_149: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    
    # No stacktrace found for following nodes
    mm_default_112: "f32[128, 1024]" = torch.ops.aten.mm.default(view_275, permute_149);  view_275 = permute_149 = None
    add_tensor_112: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_112, arg216_1);  mm_default_112 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_276: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_112, [1, 128, 1024]);  add_tensor_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_95: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_92, view_276);  add_92 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_41: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_95, getitem_55);  getitem_55 = None
    add_96: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    mul_106: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
    mul_107: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_106, arg217_1);  mul_106 = arg217_1 = None
    add_97: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_107, arg218_1);  mul_107 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_277: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_97, [128, 1024]);  add_97 = None
    permute_150: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    
    # No stacktrace found for following nodes
    mm_default_111: "f32[128, 4096]" = torch.ops.aten.mm.default(view_277, permute_150);  view_277 = permute_150 = None
    add_tensor_111: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_111, arg220_1);  mm_default_111 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_278: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_111, [1, 128, 4096]);  add_tensor_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_108: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_278, 0.5)
    mul_109: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_278, 0.7071067811865476);  view_278 = None
    erf_12: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_109);  mul_109 = None
    add_98: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_110: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_108, add_98);  mul_108 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_279: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_110, [128, 4096]);  mul_110 = None
    permute_151: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
    
    # No stacktrace found for following nodes
    mm_default_110: "f32[128, 1024]" = torch.ops.aten.mm.default(view_279, permute_151);  view_279 = permute_151 = None
    add_tensor_110: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_110, arg222_1);  mm_default_110 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_280: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_110, [1, 128, 1024]);  add_tensor_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_99: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_95, view_280);  add_95 = view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_42: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_99, getitem_57);  getitem_57 = None
    add_100: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_111: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
    mul_112: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_111, arg223_1);  mul_111 = arg223_1 = None
    add_101: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_112, arg224_1);  mul_112 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_281: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_101, [128, 1024])
    permute_152: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    
    # No stacktrace found for following nodes
    mm_default_109: "f32[128, 1024]" = torch.ops.aten.mm.default(view_281, permute_152);  view_281 = permute_152 = None
    add_tensor_109: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_109, arg226_1);  mm_default_109 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_282: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_109, [1, 128, 1024]);  add_tensor_109 = None
    mul_113: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_282, 0.125);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_289: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_113, [1, 128, 16, 64]);  mul_113 = None
    permute_157: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    clone_114: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_290: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_114, [16, -1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_283: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_101, [128, 1024])
    permute_153: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    
    # No stacktrace found for following nodes
    mm_default_108: "f32[128, 1024]" = torch.ops.aten.mm.default(view_283, permute_153);  view_283 = permute_153 = None
    add_tensor_108: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_108, arg228_1);  mm_default_108 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_284: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_108, [1, 128, 1024]);  add_tensor_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_285: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_284, [1, -1, 16, 64]);  view_284 = None
    permute_154: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_112: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_291: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_112, [16, -1, 64]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_158: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_290, permute_158);  view_290 = permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_293: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    add_102: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_293, expand_1);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_294: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_102, [16, 128, 128]);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_294, [-1], True)
    sub_43: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_294, amax_14);  view_294 = amax_14 = None
    exp_14: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_286: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_101, [128, 1024]);  add_101 = None
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
    
    # No stacktrace found for following nodes
    mm_default_107: "f32[128, 1024]" = torch.ops.aten.mm.default(view_286, permute_155);  view_286 = permute_155 = None
    add_tensor_107: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_107, arg230_1);  mm_default_107 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_287: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_107, [1, 128, 1024]);  add_tensor_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_288: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_287, [1, -1, 16, 64]);  view_287 = None
    permute_156: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    clone_113: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_292: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_113, [16, -1, 64]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_14, view_292);  div_14 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_295: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_159: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_116: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    view_296: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_116, [1, 128, 1024]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_297: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_296, [128, 1024]);  view_296 = None
    permute_160: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
    
    # No stacktrace found for following nodes
    mm_default_106: "f32[128, 1024]" = torch.ops.aten.mm.default(view_297, permute_160);  view_297 = permute_160 = None
    add_tensor_106: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_106, arg232_1);  mm_default_106 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_298: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_106, [1, 128, 1024]);  add_tensor_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_103: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_99, view_298);  add_99 = view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_44: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_103, getitem_59);  getitem_59 = None
    add_104: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    mul_114: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_115: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_114, arg233_1);  mul_114 = arg233_1 = None
    add_105: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_115, arg234_1);  mul_115 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_299: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_105, [128, 1024]);  add_105 = None
    permute_161: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
    
    # No stacktrace found for following nodes
    mm_default_105: "f32[128, 1024]" = torch.ops.aten.mm.default(view_299, permute_161);  view_299 = permute_161 = None
    add_tensor_105: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_105, arg236_1);  mm_default_105 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_300: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_105, [1, 128, 1024]);  add_tensor_105 = None
    mul_116: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_300, 0.125);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_307: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_116, [1, 128, 16, 64]);  mul_116 = None
    permute_166: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    clone_120: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_308: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_120, [16, -1, 64]);  clone_120 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_30: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_308, 0);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_301: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    
    # No stacktrace found for following nodes
    mm_default_104: "f32[128, 1024]" = torch.ops.aten.mm.default(view_301, permute_162);  view_301 = permute_162 = None
    add_tensor_104: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_104, arg238_1);  mm_default_104 = arg238_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_302: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_104, [1, 128, 1024]);  add_tensor_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_303: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_302, [1, -1, 16, 64]);  view_302 = None
    permute_163: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    clone_118: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_309: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_118, [16, -1, 64]);  clone_118 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_31: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_309, 0);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_304: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_164: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    
    # No stacktrace found for following nodes
    mm_default_103: "f32[128, 1024]" = torch.ops.aten.mm.default(view_304, permute_164);  view_304 = permute_164 = None
    add_tensor_103: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_103, arg240_1);  mm_default_103 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_305: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_103, [1, 128, 1024]);  add_tensor_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_306: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_305, [1, -1, 16, 64]);  view_305 = None
    permute_165: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
    clone_119: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_310: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_119, [16, -1, 64]);  clone_119 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_32: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_310, 0);  view_310 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, None, True, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
    getitem_134: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
    squeeze_dim_10: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_134, 0);  getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_311: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_10, [1, 16, 128, 64]);  squeeze_dim_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_168: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_122: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_312: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1024]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_313: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_312, [128, 1024]);  view_312 = None
    permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
    
    # No stacktrace found for following nodes
    mm_default_102: "f32[128, 1024]" = torch.ops.aten.mm.default(view_313, permute_169);  view_313 = permute_169 = None
    add_tensor_102: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_102, arg242_1);  mm_default_102 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_314: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_102, [1, 128, 1024]);  add_tensor_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_106: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_103, view_314);  add_103 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_46: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_106, getitem_61);  getitem_61 = None
    add_107: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    mul_117: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
    mul_118: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_117, arg243_1);  mul_117 = arg243_1 = None
    add_108: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_118, arg244_1);  mul_118 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_315: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_108, [128, 1024]);  add_108 = None
    permute_170: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    
    # No stacktrace found for following nodes
    mm_default_101: "f32[128, 4096]" = torch.ops.aten.mm.default(view_315, permute_170);  view_315 = permute_170 = None
    add_tensor_101: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_101, arg246_1);  mm_default_101 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_316: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_101, [1, 128, 4096]);  add_tensor_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_119: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_316, 0.5)
    mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_316, 0.7071067811865476);  view_316 = None
    erf_13: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_109: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_119, add_109);  mul_119 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_317: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_121, [128, 4096]);  mul_121 = None
    permute_171: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    
    # No stacktrace found for following nodes
    mm_default_100: "f32[128, 1024]" = torch.ops.aten.mm.default(view_317, permute_171);  view_317 = permute_171 = None
    add_tensor_100: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_100, arg248_1);  mm_default_100 = arg248_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_318: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_100, [1, 128, 1024]);  add_tensor_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_110: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_106, view_318);  add_106 = view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    sub_47: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_110, getitem_63);  getitem_63 = None
    add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_122: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_123: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_122, arg249_1);  mul_122 = arg249_1 = None
    add_112: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_123, arg250_1);  mul_123 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_319: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_112, [128, 1024])
    permute_172: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
    
    # No stacktrace found for following nodes
    mm_default_99: "f32[128, 1024]" = torch.ops.aten.mm.default(view_319, permute_172);  view_319 = permute_172 = None
    add_tensor_99: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_99, arg252_1);  mm_default_99 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_320: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_99, [1, 128, 1024]);  add_tensor_99 = None
    mul_124: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_320, 0.125);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_327: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_124, [1, 128, 16, 64]);  mul_124 = None
    permute_177: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    clone_128: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_328: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_128, [16, -1, 64]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_321: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_112, [128, 1024])
    permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    
    # No stacktrace found for following nodes
    mm_default_98: "f32[128, 1024]" = torch.ops.aten.mm.default(view_321, permute_173);  view_321 = permute_173 = None
    add_tensor_98: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_98, arg254_1);  mm_default_98 = arg254_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_322: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_98, [1, 128, 1024]);  add_tensor_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_323: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_322, [1, -1, 16, 64]);  view_322 = None
    permute_174: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
    clone_126: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_329: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_126, [16, -1, 64]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_178: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_328, permute_178);  view_328 = permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_331: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    add_113: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_331, expand_1);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_332: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_113, [16, 128, 128]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_332, [-1], True)
    sub_48: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_332, amax_16);  view_332 = amax_16 = None
    exp_16: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_17: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_324: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_112, [128, 1024]);  add_112 = None
    permute_175: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
    
    # No stacktrace found for following nodes
    mm_default_97: "f32[128, 1024]" = torch.ops.aten.mm.default(view_324, permute_175);  view_324 = permute_175 = None
    add_tensor_97: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_97, arg256_1);  mm_default_97 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_325: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_97, [1, 128, 1024]);  add_tensor_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_326: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_325, [1, -1, 16, 64]);  view_325 = None
    permute_176: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
    clone_127: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_330: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_127, [16, -1, 64]);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_16, view_330);  div_16 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_333: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_179: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_130: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_334: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1024]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_335: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_334, [128, 1024]);  view_334 = None
    permute_180: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
    
    # No stacktrace found for following nodes
    mm_default_96: "f32[128, 1024]" = torch.ops.aten.mm.default(view_335, permute_180);  view_335 = permute_180 = None
    add_tensor_96: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_96, arg258_1);  mm_default_96 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_336: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_96, [1, 128, 1024]);  add_tensor_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_114: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_110, view_336);  add_110 = view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    sub_49: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_114, getitem_65);  getitem_65 = None
    add_115: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    mul_125: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
    mul_126: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_125, arg259_1);  mul_125 = arg259_1 = None
    add_116: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_126, arg260_1);  mul_126 = arg260_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_337: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_116, [128, 1024]);  add_116 = None
    permute_181: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    
    # No stacktrace found for following nodes
    mm_default_95: "f32[128, 1024]" = torch.ops.aten.mm.default(view_337, permute_181);  view_337 = permute_181 = None
    add_tensor_95: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_95, arg262_1);  mm_default_95 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_338: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_95, [1, 128, 1024]);  add_tensor_95 = None
    mul_127: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_338, 0.125);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_345: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_127, [1, 128, 16, 64]);  mul_127 = None
    permute_186: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_134: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_346: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_134, [16, -1, 64]);  clone_134 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_27: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_346, 0);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_339: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_182: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
    
    # No stacktrace found for following nodes
    mm_default_94: "f32[128, 1024]" = torch.ops.aten.mm.default(view_339, permute_182);  view_339 = permute_182 = None
    add_tensor_94: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_94, arg264_1);  mm_default_94 = arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_340: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_94, [1, 128, 1024]);  add_tensor_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_341: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_340, [1, -1, 16, 64]);  view_340 = None
    permute_183: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    clone_132: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_347: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_132, [16, -1, 64]);  clone_132 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_28: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_347, 0);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_342: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
    
    # No stacktrace found for following nodes
    mm_default_93: "f32[128, 1024]" = torch.ops.aten.mm.default(view_342, permute_184);  view_342 = permute_184 = None
    add_tensor_93: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_93, arg266_1);  mm_default_93 = arg266_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_343: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_93, [1, 128, 1024]);  add_tensor_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_344: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_343, [1, -1, 16, 64]);  view_343 = None
    permute_185: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    clone_133: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_348: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_133, [16, -1, 64]);  clone_133 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_29: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_348, 0);  view_348 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, None, True, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
    getitem_133: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
    squeeze_dim_9: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_133, 0);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_349: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_9, [1, 16, 128, 64]);  squeeze_dim_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_188: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_136: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_350: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_136, [1, 128, 1024]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_351: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_350, [128, 1024]);  view_350 = None
    permute_189: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    
    # No stacktrace found for following nodes
    mm_default_92: "f32[128, 1024]" = torch.ops.aten.mm.default(view_351, permute_189);  view_351 = permute_189 = None
    add_tensor_92: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_92, arg268_1);  mm_default_92 = arg268_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_352: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_92, [1, 128, 1024]);  add_tensor_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_117: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_114, view_352);  add_114 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    sub_51: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_67);  getitem_67 = None
    add_118: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    mul_128: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = rsqrt_33 = None
    mul_129: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_128, arg269_1);  mul_128 = arg269_1 = None
    add_119: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_129, arg270_1);  mul_129 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_353: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_119, [128, 1024]);  add_119 = None
    permute_190: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
    
    # No stacktrace found for following nodes
    mm_default_91: "f32[128, 4096]" = torch.ops.aten.mm.default(view_353, permute_190);  view_353 = permute_190 = None
    add_tensor_91: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_91, arg272_1);  mm_default_91 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_354: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_91, [1, 128, 4096]);  add_tensor_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_130: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_354, 0.5)
    mul_131: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_354, 0.7071067811865476);  view_354 = None
    erf_14: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_120: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_132: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, add_120);  mul_130 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_355: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_132, [128, 4096]);  mul_132 = None
    permute_191: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
    
    # No stacktrace found for following nodes
    mm_default_90: "f32[128, 1024]" = torch.ops.aten.mm.default(view_355, permute_191);  view_355 = permute_191 = None
    add_tensor_90: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_90, arg274_1);  mm_default_90 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_356: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_90, [1, 128, 1024]);  add_tensor_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_121: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_117, view_356);  add_117 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    sub_52: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_121, getitem_69);  getitem_69 = None
    add_122: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    mul_133: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
    mul_134: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_133, arg275_1);  mul_133 = arg275_1 = None
    add_123: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_134, arg276_1);  mul_134 = arg276_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_357: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_123, [128, 1024])
    permute_192: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    
    # No stacktrace found for following nodes
    mm_default_89: "f32[128, 1024]" = torch.ops.aten.mm.default(view_357, permute_192);  view_357 = permute_192 = None
    add_tensor_89: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_89, arg278_1);  mm_default_89 = arg278_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_358: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_89, [1, 128, 1024]);  add_tensor_89 = None
    mul_135: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_358, 0.125);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_365: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_135, [1, 128, 16, 64]);  mul_135 = None
    permute_197: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    clone_142: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_366: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_142, [16, -1, 64]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_359: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_123, [128, 1024])
    permute_193: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
    
    # No stacktrace found for following nodes
    mm_default_88: "f32[128, 1024]" = torch.ops.aten.mm.default(view_359, permute_193);  view_359 = permute_193 = None
    add_tensor_88: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_88, arg280_1);  mm_default_88 = arg280_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_360: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_88, [1, 128, 1024]);  add_tensor_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_361: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_360, [1, -1, 16, 64]);  view_360 = None
    permute_194: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    clone_140: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_367: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_140, [16, -1, 64]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_198: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_366, permute_198);  view_366 = permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_369: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    add_124: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_369, expand_1);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_370: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_124, [16, 128, 128]);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_370, [-1], True)
    sub_53: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_370, amax_18);  view_370 = amax_18 = None
    exp_18: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
    sum_19: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_362: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_123, [128, 1024]);  add_123 = None
    permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
    
    # No stacktrace found for following nodes
    mm_default_87: "f32[128, 1024]" = torch.ops.aten.mm.default(view_362, permute_195);  view_362 = permute_195 = None
    add_tensor_87: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_87, arg282_1);  mm_default_87 = arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_363: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_87, [1, 128, 1024]);  add_tensor_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_364: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_363, [1, -1, 16, 64]);  view_363 = None
    permute_196: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    clone_141: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_368: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_141, [16, -1, 64]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_37: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_18, view_368);  div_18 = view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_371: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_199: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_144: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_372: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_144, [1, 128, 1024]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_373: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_372, [128, 1024]);  view_372 = None
    permute_200: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
    
    # No stacktrace found for following nodes
    mm_default_86: "f32[128, 1024]" = torch.ops.aten.mm.default(view_373, permute_200);  view_373 = permute_200 = None
    add_tensor_86: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_86, arg284_1);  mm_default_86 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_374: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_86, [1, 128, 1024]);  add_tensor_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_125: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_121, view_374);  add_121 = view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    sub_54: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_125, getitem_71);  getitem_71 = None
    add_126: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    mul_136: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = rsqrt_35 = None
    mul_137: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_136, arg285_1);  mul_136 = arg285_1 = None
    add_127: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_137, arg286_1);  mul_137 = arg286_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_375: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_127, [128, 1024]);  add_127 = None
    permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
    
    # No stacktrace found for following nodes
    mm_default_85: "f32[128, 1024]" = torch.ops.aten.mm.default(view_375, permute_201);  view_375 = permute_201 = None
    add_tensor_85: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_85, arg288_1);  mm_default_85 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_376: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_85, [1, 128, 1024]);  add_tensor_85 = None
    mul_138: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_376, 0.125);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_383: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_138, [1, 128, 16, 64]);  mul_138 = None
    permute_206: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    clone_148: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_384: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_148, [16, -1, 64]);  clone_148 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_24: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_384, 0);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_377: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_202: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
    
    # No stacktrace found for following nodes
    mm_default_84: "f32[128, 1024]" = torch.ops.aten.mm.default(view_377, permute_202);  view_377 = permute_202 = None
    add_tensor_84: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_84, arg290_1);  mm_default_84 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_378: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_84, [1, 128, 1024]);  add_tensor_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_379: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_378, [1, -1, 16, 64]);  view_378 = None
    permute_203: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
    clone_146: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_385: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_146, [16, -1, 64]);  clone_146 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_25: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_385, 0);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_380: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_204: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
    
    # No stacktrace found for following nodes
    mm_default_83: "f32[128, 1024]" = torch.ops.aten.mm.default(view_380, permute_204);  view_380 = permute_204 = None
    add_tensor_83: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_83, arg292_1);  mm_default_83 = arg292_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_381: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_83, [1, 128, 1024]);  add_tensor_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_382: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_381, [1, -1, 16, 64]);  view_381 = None
    permute_205: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    clone_147: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_386: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_147, [16, -1, 64]);  clone_147 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_26: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_386, 0);  view_386 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, None, True, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
    getitem_132: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
    squeeze_dim_8: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_132, 0);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_387: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_8, [1, 16, 128, 64]);  squeeze_dim_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_208: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_150: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    view_388: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_150, [1, 128, 1024]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_389: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_388, [128, 1024]);  view_388 = None
    permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    
    # No stacktrace found for following nodes
    mm_default_82: "f32[128, 1024]" = torch.ops.aten.mm.default(view_389, permute_209);  view_389 = permute_209 = None
    add_tensor_82: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_82, arg294_1);  mm_default_82 = arg294_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_390: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_82, [1, 128, 1024]);  add_tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_128: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_125, view_390);  add_125 = view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    sub_56: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_128, getitem_73);  getitem_73 = None
    add_129: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    mul_139: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_36);  sub_56 = rsqrt_36 = None
    mul_140: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_139, arg295_1);  mul_139 = arg295_1 = None
    add_130: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_140, arg296_1);  mul_140 = arg296_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_391: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_130, [128, 1024]);  add_130 = None
    permute_210: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
    
    # No stacktrace found for following nodes
    mm_default_81: "f32[128, 4096]" = torch.ops.aten.mm.default(view_391, permute_210);  view_391 = permute_210 = None
    add_tensor_81: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_81, arg298_1);  mm_default_81 = arg298_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_392: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_81, [1, 128, 4096]);  add_tensor_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_141: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_392, 0.5)
    mul_142: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_392, 0.7071067811865476);  view_392 = None
    erf_15: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
    add_131: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_143: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_141, add_131);  mul_141 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_393: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_143, [128, 4096]);  mul_143 = None
    permute_211: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
    
    # No stacktrace found for following nodes
    mm_default_80: "f32[128, 1024]" = torch.ops.aten.mm.default(view_393, permute_211);  view_393 = permute_211 = None
    add_tensor_80: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_80, arg300_1);  mm_default_80 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_394: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_80, [1, 128, 1024]);  add_tensor_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_132: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_128, view_394);  add_128 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    sub_57: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_132, getitem_75);  getitem_75 = None
    add_133: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    mul_144: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = rsqrt_37 = None
    mul_145: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_144, arg301_1);  mul_144 = arg301_1 = None
    add_134: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_145, arg302_1);  mul_145 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_395: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_134, [128, 1024])
    permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
    
    # No stacktrace found for following nodes
    mm_default_79: "f32[128, 1024]" = torch.ops.aten.mm.default(view_395, permute_212);  view_395 = permute_212 = None
    add_tensor_79: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_79, arg304_1);  mm_default_79 = arg304_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_396: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_79, [1, 128, 1024]);  add_tensor_79 = None
    mul_146: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_396, 0.125);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_403: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_146, [1, 128, 16, 64]);  mul_146 = None
    permute_217: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    clone_156: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_404: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_156, [16, -1, 64]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_397: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_134, [128, 1024])
    permute_213: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
    
    # No stacktrace found for following nodes
    mm_default_78: "f32[128, 1024]" = torch.ops.aten.mm.default(view_397, permute_213);  view_397 = permute_213 = None
    add_tensor_78: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_78, arg306_1);  mm_default_78 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_398: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_78, [1, 128, 1024]);  add_tensor_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_399: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_398, [1, -1, 16, 64]);  view_398 = None
    permute_214: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
    clone_154: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_405: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_154, [16, -1, 64]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_218: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_404, permute_218);  view_404 = permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_407: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    add_135: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_407, expand_1);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_408: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_135, [16, 128, 128]);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_408, [-1], True)
    sub_58: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_408, amax_20);  view_408 = amax_20 = None
    exp_20: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_21: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_400: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_134, [128, 1024]);  add_134 = None
    permute_215: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
    
    # No stacktrace found for following nodes
    mm_default_77: "f32[128, 1024]" = torch.ops.aten.mm.default(view_400, permute_215);  view_400 = permute_215 = None
    add_tensor_77: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_77, arg308_1);  mm_default_77 = arg308_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_401: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_77, [1, 128, 1024]);  add_tensor_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_402: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_401, [1, -1, 16, 64]);  view_401 = None
    permute_216: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    clone_155: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_406: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_155, [16, -1, 64]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_41: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_20, view_406);  div_20 = view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_409: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_219: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_158: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_410: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_158, [1, 128, 1024]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_411: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_410, [128, 1024]);  view_410 = None
    permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
    
    # No stacktrace found for following nodes
    mm_default_76: "f32[128, 1024]" = torch.ops.aten.mm.default(view_411, permute_220);  view_411 = permute_220 = None
    add_tensor_76: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_76, arg310_1);  mm_default_76 = arg310_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_412: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_76, [1, 128, 1024]);  add_tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_136: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_132, view_412);  add_132 = view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    sub_59: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_136, getitem_77);  getitem_77 = None
    add_137: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    mul_147: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_38);  sub_59 = rsqrt_38 = None
    mul_148: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_147, arg311_1);  mul_147 = arg311_1 = None
    add_138: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_148, arg312_1);  mul_148 = arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_413: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_138, [128, 1024]);  add_138 = None
    permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
    
    # No stacktrace found for following nodes
    mm_default_75: "f32[128, 1024]" = torch.ops.aten.mm.default(view_413, permute_221);  view_413 = permute_221 = None
    add_tensor_75: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_75, arg314_1);  mm_default_75 = arg314_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_414: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_75, [1, 128, 1024]);  add_tensor_75 = None
    mul_149: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_414, 0.125);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_421: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_149, [1, 128, 16, 64]);  mul_149 = None
    permute_226: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    clone_162: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_422: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_162, [16, -1, 64]);  clone_162 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_21: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_422, 0);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_415: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_222: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
    
    # No stacktrace found for following nodes
    mm_default_74: "f32[128, 1024]" = torch.ops.aten.mm.default(view_415, permute_222);  view_415 = permute_222 = None
    add_tensor_74: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_74, arg316_1);  mm_default_74 = arg316_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_416: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_74, [1, 128, 1024]);  add_tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_417: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_416, [1, -1, 16, 64]);  view_416 = None
    permute_223: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    clone_160: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_423: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_160, [16, -1, 64]);  clone_160 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_22: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_423, 0);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_418: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_224: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
    
    # No stacktrace found for following nodes
    mm_default_73: "f32[128, 1024]" = torch.ops.aten.mm.default(view_418, permute_224);  view_418 = permute_224 = None
    add_tensor_73: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_73, arg318_1);  mm_default_73 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_419: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_73, [1, 128, 1024]);  add_tensor_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_420: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_419, [1, -1, 16, 64]);  view_419 = None
    permute_225: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    clone_161: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_424: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_161, [16, -1, 64]);  clone_161 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_23: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_424, 0);  view_424 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, None, True, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
    getitem_131: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
    squeeze_dim_7: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_131, 0);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_425: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_7, [1, 16, 128, 64]);  squeeze_dim_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_228: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_164: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_426: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_164, [1, 128, 1024]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_427: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_426, [128, 1024]);  view_426 = None
    permute_229: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
    
    # No stacktrace found for following nodes
    mm_default_72: "f32[128, 1024]" = torch.ops.aten.mm.default(view_427, permute_229);  view_427 = permute_229 = None
    add_tensor_72: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_72, arg320_1);  mm_default_72 = arg320_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_428: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_72, [1, 128, 1024]);  add_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_139: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_136, view_428);  add_136 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    sub_61: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_139, getitem_79);  getitem_79 = None
    add_140: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    mul_150: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_39);  sub_61 = rsqrt_39 = None
    mul_151: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_150, arg321_1);  mul_150 = arg321_1 = None
    add_141: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_151, arg322_1);  mul_151 = arg322_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_429: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_141, [128, 1024]);  add_141 = None
    permute_230: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[128, 4096]" = torch.ops.aten.mm.default(view_429, permute_230);  view_429 = permute_230 = None
    add_tensor_71: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_71, arg324_1);  mm_default_71 = arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_430: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_71, [1, 128, 4096]);  add_tensor_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_152: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_430, 0.5)
    mul_153: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_430, 0.7071067811865476);  view_430 = None
    erf_16: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_142: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_154: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_152, add_142);  mul_152 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_431: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_154, [128, 4096]);  mul_154 = None
    permute_231: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[128, 1024]" = torch.ops.aten.mm.default(view_431, permute_231);  view_431 = permute_231 = None
    add_tensor_70: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_70, arg326_1);  mm_default_70 = arg326_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_432: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_70, [1, 128, 1024]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_143: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_139, view_432);  add_139 = view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    sub_62: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_143, getitem_81);  getitem_81 = None
    add_144: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    mul_155: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_40);  sub_62 = rsqrt_40 = None
    mul_156: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_155, arg327_1);  mul_155 = arg327_1 = None
    add_145: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_156, arg328_1);  mul_156 = arg328_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_433: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_145, [128, 1024])
    permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[128, 1024]" = torch.ops.aten.mm.default(view_433, permute_232);  view_433 = permute_232 = None
    add_tensor_69: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_69, arg330_1);  mm_default_69 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_434: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_69, [1, 128, 1024]);  add_tensor_69 = None
    mul_157: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_434, 0.125);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_441: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_157, [1, 128, 16, 64]);  mul_157 = None
    permute_237: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
    clone_170: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_442: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_170, [16, -1, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_435: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_145, [128, 1024])
    permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[128, 1024]" = torch.ops.aten.mm.default(view_435, permute_233);  view_435 = permute_233 = None
    add_tensor_68: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_68, arg332_1);  mm_default_68 = arg332_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_436: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_68, [1, 128, 1024]);  add_tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_437: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_436, [1, -1, 16, 64]);  view_436 = None
    permute_234: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    clone_168: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_443: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_168, [16, -1, 64]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_238: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_443, [0, 2, 1]);  view_443 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_442, permute_238);  view_442 = permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_445: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    add_146: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_445, expand_1);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_446: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_146, [16, 128, 128]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_446, [-1], True)
    sub_63: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_446, amax_22);  view_446 = amax_22 = None
    exp_22: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_23: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_438: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_145, [128, 1024]);  add_145 = None
    permute_235: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[128, 1024]" = torch.ops.aten.mm.default(view_438, permute_235);  view_438 = permute_235 = None
    add_tensor_67: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_67, arg334_1);  mm_default_67 = arg334_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_439: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_67, [1, 128, 1024]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_440: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_439, [1, -1, 16, 64]);  view_439 = None
    permute_236: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
    clone_169: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_444: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_169, [16, -1, 64]);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_45: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_22, view_444);  div_22 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_447: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_239: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_172: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
    view_448: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_172, [1, 128, 1024]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_449: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_448, [128, 1024]);  view_448 = None
    permute_240: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[128, 1024]" = torch.ops.aten.mm.default(view_449, permute_240);  view_449 = permute_240 = None
    add_tensor_66: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_66, arg336_1);  mm_default_66 = arg336_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_450: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_66, [1, 128, 1024]);  add_tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_147: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_143, view_450);  add_143 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    sub_64: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_147, getitem_83);  getitem_83 = None
    add_148: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    mul_158: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_41);  sub_64 = rsqrt_41 = None
    mul_159: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_158, arg337_1);  mul_158 = arg337_1 = None
    add_149: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_159, arg338_1);  mul_159 = arg338_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_451: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_149, [128, 1024]);  add_149 = None
    permute_241: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[128, 1024]" = torch.ops.aten.mm.default(view_451, permute_241);  view_451 = permute_241 = None
    add_tensor_65: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_65, arg340_1);  mm_default_65 = arg340_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_452: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_65, [1, 128, 1024]);  add_tensor_65 = None
    mul_160: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_452, 0.125);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_459: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_160, [1, 128, 16, 64]);  mul_160 = None
    permute_246: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    clone_176: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_460: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_176, [16, -1, 64]);  clone_176 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_18: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_460, 0);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_453: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[128, 1024]" = torch.ops.aten.mm.default(view_453, permute_242);  view_453 = permute_242 = None
    add_tensor_64: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_64, arg342_1);  mm_default_64 = arg342_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_454: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_64, [1, 128, 1024]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_455: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_454, [1, -1, 16, 64]);  view_454 = None
    permute_243: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
    clone_174: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_461: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_174, [16, -1, 64]);  clone_174 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_19: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_461, 0);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_456: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_244: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[128, 1024]" = torch.ops.aten.mm.default(view_456, permute_244);  view_456 = permute_244 = None
    add_tensor_63: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_63, arg344_1);  mm_default_63 = arg344_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_457: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_63, [1, 128, 1024]);  add_tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_458: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_457, [1, -1, 16, 64]);  view_457 = None
    permute_245: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
    clone_175: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_462: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_175, [16, -1, 64]);  clone_175 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_20: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_462, 0);  view_462 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, None, True, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
    getitem_130: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
    squeeze_dim_6: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_130, 0);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_463: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_6, [1, 16, 128, 64]);  squeeze_dim_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_248: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_178: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    view_464: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1024]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_465: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_464, [128, 1024]);  view_464 = None
    permute_249: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[128, 1024]" = torch.ops.aten.mm.default(view_465, permute_249);  view_465 = permute_249 = None
    add_tensor_62: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_62, arg346_1);  mm_default_62 = arg346_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_466: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_62, [1, 128, 1024]);  add_tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_150: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_147, view_466);  add_147 = view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    sub_66: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_150, getitem_85);  getitem_85 = None
    add_151: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    mul_161: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_42);  sub_66 = rsqrt_42 = None
    mul_162: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_161, arg347_1);  mul_161 = arg347_1 = None
    add_152: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_162, arg348_1);  mul_162 = arg348_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_467: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_152, [128, 1024]);  add_152 = None
    permute_250: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[128, 4096]" = torch.ops.aten.mm.default(view_467, permute_250);  view_467 = permute_250 = None
    add_tensor_61: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_61, arg350_1);  mm_default_61 = arg350_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_468: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_61, [1, 128, 4096]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_163: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_468, 0.5)
    mul_164: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_468, 0.7071067811865476);  view_468 = None
    erf_17: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_153: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_165: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_163, add_153);  mul_163 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_469: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_165, [128, 4096]);  mul_165 = None
    permute_251: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[128, 1024]" = torch.ops.aten.mm.default(view_469, permute_251);  view_469 = permute_251 = None
    add_tensor_60: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_60, arg352_1);  mm_default_60 = arg352_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_470: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_60, [1, 128, 1024]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_154: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_150, view_470);  add_150 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    sub_67: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_154, getitem_87);  getitem_87 = None
    add_155: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    mul_166: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_43);  sub_67 = rsqrt_43 = None
    mul_167: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_166, arg353_1);  mul_166 = arg353_1 = None
    add_156: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_167, arg354_1);  mul_167 = arg354_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_471: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_156, [128, 1024])
    permute_252: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[128, 1024]" = torch.ops.aten.mm.default(view_471, permute_252);  view_471 = permute_252 = None
    add_tensor_59: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_59, arg356_1);  mm_default_59 = arg356_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_472: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_59, [1, 128, 1024]);  add_tensor_59 = None
    mul_168: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_472, 0.125);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_479: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_168, [1, 128, 16, 64]);  mul_168 = None
    permute_257: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
    clone_184: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_480: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_184, [16, -1, 64]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_473: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_156, [128, 1024])
    permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[128, 1024]" = torch.ops.aten.mm.default(view_473, permute_253);  view_473 = permute_253 = None
    add_tensor_58: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_58, arg358_1);  mm_default_58 = arg358_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_474: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_58, [1, 128, 1024]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_475: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_474, [1, -1, 16, 64]);  view_474 = None
    permute_254: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    clone_182: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_481: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_182, [16, -1, 64]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_258: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_481, [0, 2, 1]);  view_481 = None
    bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_480, permute_258);  view_480 = permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_483: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None
    add_157: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_483, expand_1);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_484: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_157, [16, 128, 128]);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_24: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_484, [-1], True)
    sub_68: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_484, amax_24);  view_484 = amax_24 = None
    exp_24: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_25: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_24: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_476: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_156, [128, 1024]);  add_156 = None
    permute_255: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[128, 1024]" = torch.ops.aten.mm.default(view_476, permute_255);  view_476 = permute_255 = None
    add_tensor_57: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_57, arg360_1);  mm_default_57 = arg360_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_477: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_57, [1, 128, 1024]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_478: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_477, [1, -1, 16, 64]);  view_477 = None
    permute_256: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
    clone_183: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_482: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_183, [16, -1, 64]);  clone_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_49: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_24, view_482);  div_24 = view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_485: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_49, [1, 16, 128, 64]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_259: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_485, [0, 2, 1, 3]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_186: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_486: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1024]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_487: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_486, [128, 1024]);  view_486 = None
    permute_260: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[128, 1024]" = torch.ops.aten.mm.default(view_487, permute_260);  view_487 = permute_260 = None
    add_tensor_56: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_56, arg362_1);  mm_default_56 = arg362_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_488: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_56, [1, 128, 1024]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_158: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_154, view_488);  add_154 = view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    sub_69: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_158, getitem_89);  getitem_89 = None
    add_159: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    mul_169: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_44);  sub_69 = rsqrt_44 = None
    mul_170: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_169, arg363_1);  mul_169 = arg363_1 = None
    add_160: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_170, arg364_1);  mul_170 = arg364_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_489: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_160, [128, 1024]);  add_160 = None
    permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg365_1, [1, 0]);  arg365_1 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[128, 1024]" = torch.ops.aten.mm.default(view_489, permute_261);  view_489 = permute_261 = None
    add_tensor_55: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_55, arg366_1);  mm_default_55 = arg366_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_490: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_55, [1, 128, 1024]);  add_tensor_55 = None
    mul_171: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_490, 0.125);  view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_497: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_171, [1, 128, 16, 64]);  mul_171 = None
    permute_266: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_190: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_498: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_190, [16, -1, 64]);  clone_190 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_15: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_498, 0);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_491: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_262: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[128, 1024]" = torch.ops.aten.mm.default(view_491, permute_262);  view_491 = permute_262 = None
    add_tensor_54: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_54, arg368_1);  mm_default_54 = arg368_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_492: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_54, [1, 128, 1024]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_493: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_492, [1, -1, 16, 64]);  view_492 = None
    permute_263: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    clone_188: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_499: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_188, [16, -1, 64]);  clone_188 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_16: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_499, 0);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_494: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[128, 1024]" = torch.ops.aten.mm.default(view_494, permute_264);  view_494 = permute_264 = None
    add_tensor_53: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_53, arg370_1);  mm_default_53 = arg370_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_495: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_53, [1, 128, 1024]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_496: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_495, [1, -1, 16, 64]);  view_495 = None
    permute_265: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
    clone_189: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_500: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_189, [16, -1, 64]);  clone_189 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_17: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_500, 0);  view_500 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, None, True, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
    getitem_129: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
    squeeze_dim_5: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_129, 0);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_501: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_5, [1, 16, 128, 64]);  squeeze_dim_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_268: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_192: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    view_502: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_192, [1, 128, 1024]);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_503: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_502, [128, 1024]);  view_502 = None
    permute_269: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[128, 1024]" = torch.ops.aten.mm.default(view_503, permute_269);  view_503 = permute_269 = None
    add_tensor_52: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_52, arg372_1);  mm_default_52 = arg372_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_504: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_52, [1, 128, 1024]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_161: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_158, view_504);  add_158 = view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    sub_71: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_161, getitem_91);  getitem_91 = None
    add_162: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    mul_172: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_45);  sub_71 = rsqrt_45 = None
    mul_173: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_172, arg373_1);  mul_172 = arg373_1 = None
    add_163: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_173, arg374_1);  mul_173 = arg374_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_505: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_163, [128, 1024]);  add_163 = None
    permute_270: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[128, 4096]" = torch.ops.aten.mm.default(view_505, permute_270);  view_505 = permute_270 = None
    add_tensor_51: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_51, arg376_1);  mm_default_51 = arg376_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_506: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_51, [1, 128, 4096]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_174: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_506, 0.5)
    mul_175: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_506, 0.7071067811865476);  view_506 = None
    erf_18: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_164: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_176: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_174, add_164);  mul_174 = add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_507: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_176, [128, 4096]);  mul_176 = None
    permute_271: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[128, 1024]" = torch.ops.aten.mm.default(view_507, permute_271);  view_507 = permute_271 = None
    add_tensor_50: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_50, arg378_1);  mm_default_50 = arg378_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_508: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_50, [1, 128, 1024]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_165: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_161, view_508);  add_161 = view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    sub_72: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_165, getitem_93);  getitem_93 = None
    add_166: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    mul_177: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_46);  sub_72 = rsqrt_46 = None
    mul_178: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_177, arg379_1);  mul_177 = arg379_1 = None
    add_167: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_178, arg380_1);  mul_178 = arg380_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_509: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_167, [128, 1024])
    permute_272: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[128, 1024]" = torch.ops.aten.mm.default(view_509, permute_272);  view_509 = permute_272 = None
    add_tensor_49: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_49, arg382_1);  mm_default_49 = arg382_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_510: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_49, [1, 128, 1024]);  add_tensor_49 = None
    mul_179: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_510, 0.125);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_517: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_179, [1, 128, 16, 64]);  mul_179 = None
    permute_277: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    clone_198: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_518: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_198, [16, -1, 64]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_511: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_167, [128, 1024])
    permute_273: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[128, 1024]" = torch.ops.aten.mm.default(view_511, permute_273);  view_511 = permute_273 = None
    add_tensor_48: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_48, arg384_1);  mm_default_48 = arg384_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_512: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_48, [1, 128, 1024]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_513: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_512, [1, -1, 16, 64]);  view_512 = None
    permute_274: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
    clone_196: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_519: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_196, [16, -1, 64]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_278: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
    bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_518, permute_278);  view_518 = permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_521: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None
    add_168: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_521, expand_1);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_522: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_168, [16, 128, 128]);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_26: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_522, [-1], True)
    sub_73: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_522, amax_26);  view_522 = amax_26 = None
    exp_26: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_27: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_26: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_514: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_167, [128, 1024]);  add_167 = None
    permute_275: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[128, 1024]" = torch.ops.aten.mm.default(view_514, permute_275);  view_514 = permute_275 = None
    add_tensor_47: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_47, arg386_1);  mm_default_47 = arg386_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_515: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_47, [1, 128, 1024]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_516: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_515, [1, -1, 16, 64]);  view_515 = None
    permute_276: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
    clone_197: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_520: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_197, [16, -1, 64]);  clone_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_53: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_26, view_520);  div_26 = view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_523: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_53, [1, 16, 128, 64]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_279: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_523, [0, 2, 1, 3]);  view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_200: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_524: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_200, [1, 128, 1024]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_525: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_524, [128, 1024]);  view_524 = None
    permute_280: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[128, 1024]" = torch.ops.aten.mm.default(view_525, permute_280);  view_525 = permute_280 = None
    add_tensor_46: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_46, arg388_1);  mm_default_46 = arg388_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_526: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_46, [1, 128, 1024]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_169: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_165, view_526);  add_165 = view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    sub_74: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_169, getitem_95);  getitem_95 = None
    add_170: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    mul_180: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_47);  sub_74 = rsqrt_47 = None
    mul_181: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_180, arg389_1);  mul_180 = arg389_1 = None
    add_171: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_181, arg390_1);  mul_181 = arg390_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_527: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_171, [128, 1024]);  add_171 = None
    permute_281: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[128, 1024]" = torch.ops.aten.mm.default(view_527, permute_281);  view_527 = permute_281 = None
    add_tensor_45: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_45, arg392_1);  mm_default_45 = arg392_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_528: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_45, [1, 128, 1024]);  add_tensor_45 = None
    mul_182: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_528, 0.125);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_535: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_182, [1, 128, 16, 64]);  mul_182 = None
    permute_286: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
    clone_204: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_536: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_204, [16, -1, 64]);  clone_204 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_12: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_536, 0);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_529: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_282: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[128, 1024]" = torch.ops.aten.mm.default(view_529, permute_282);  view_529 = permute_282 = None
    add_tensor_44: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_44, arg394_1);  mm_default_44 = arg394_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_530: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_44, [1, 128, 1024]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_531: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_530, [1, -1, 16, 64]);  view_530 = None
    permute_283: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
    clone_202: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_537: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_202, [16, -1, 64]);  clone_202 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_13: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_537, 0);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_532: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_284: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[128, 1024]" = torch.ops.aten.mm.default(view_532, permute_284);  view_532 = permute_284 = None
    add_tensor_43: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_43, arg396_1);  mm_default_43 = arg396_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_533: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_43, [1, 128, 1024]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_534: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_533, [1, -1, 16, 64]);  view_533 = None
    permute_285: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    clone_203: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_538: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_203, [16, -1, 64]);  clone_203 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_14: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_538, 0);  view_538 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, None, True, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
    getitem_128: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
    squeeze_dim_4: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_128, 0);  getitem_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_539: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_4, [1, 16, 128, 64]);  squeeze_dim_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_288: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_206: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_540: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_206, [1, 128, 1024]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_541: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_540, [128, 1024]);  view_540 = None
    permute_289: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[128, 1024]" = torch.ops.aten.mm.default(view_541, permute_289);  view_541 = permute_289 = None
    add_tensor_42: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_42, arg398_1);  mm_default_42 = arg398_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_542: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_42, [1, 128, 1024]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_172: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_169, view_542);  add_169 = view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_172, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    sub_76: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_172, getitem_97);  getitem_97 = None
    add_173: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    mul_183: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_48);  sub_76 = rsqrt_48 = None
    mul_184: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_183, arg399_1);  mul_183 = arg399_1 = None
    add_174: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_184, arg400_1);  mul_184 = arg400_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_543: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_174, [128, 1024]);  add_174 = None
    permute_290: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg401_1, [1, 0]);  arg401_1 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[128, 4096]" = torch.ops.aten.mm.default(view_543, permute_290);  view_543 = permute_290 = None
    add_tensor_41: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_41, arg402_1);  mm_default_41 = arg402_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_544: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_41, [1, 128, 4096]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_185: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_544, 0.5)
    mul_186: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476);  view_544 = None
    erf_19: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
    add_175: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_187: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_185, add_175);  mul_185 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_545: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_187, [128, 4096]);  mul_187 = None
    permute_291: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[128, 1024]" = torch.ops.aten.mm.default(view_545, permute_291);  view_545 = permute_291 = None
    add_tensor_40: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_40, arg404_1);  mm_default_40 = arg404_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_546: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_40, [1, 128, 1024]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_176: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_172, view_546);  add_172 = view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 128, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 128, 1]" = var_mean_49[1];  var_mean_49 = None
    sub_77: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_176, getitem_99);  getitem_99 = None
    add_177: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    mul_188: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_49);  sub_77 = rsqrt_49 = None
    mul_189: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_188, arg405_1);  mul_188 = arg405_1 = None
    add_178: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_189, arg406_1);  mul_189 = arg406_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_547: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_178, [128, 1024])
    permute_292: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[128, 1024]" = torch.ops.aten.mm.default(view_547, permute_292);  view_547 = permute_292 = None
    add_tensor_39: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_39, arg408_1);  mm_default_39 = arg408_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_548: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_39, [1, 128, 1024]);  add_tensor_39 = None
    mul_190: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_548, 0.125);  view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_555: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_190, [1, 128, 16, 64]);  mul_190 = None
    permute_297: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
    clone_212: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_556: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_212, [16, -1, 64]);  clone_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_549: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_178, [128, 1024])
    permute_293: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[128, 1024]" = torch.ops.aten.mm.default(view_549, permute_293);  view_549 = permute_293 = None
    add_tensor_38: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_38, arg410_1);  mm_default_38 = arg410_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_550: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_38, [1, 128, 1024]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_551: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_550, [1, -1, 16, 64]);  view_550 = None
    permute_294: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
    clone_210: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_557: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_210, [16, -1, 64]);  clone_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_298: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_557, [0, 2, 1]);  view_557 = None
    bmm_56: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_556, permute_298);  view_556 = permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_559: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_56, [1, 16, 128, 128]);  bmm_56 = None
    add_179: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_559, expand_1);  view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_560: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_179, [16, 128, 128]);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_28: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_560, [-1], True)
    sub_78: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_560, amax_28);  view_560 = amax_28 = None
    exp_28: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_78);  sub_78 = None
    sum_29: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
    div_28: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_552: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_178, [128, 1024]);  add_178 = None
    permute_295: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[128, 1024]" = torch.ops.aten.mm.default(view_552, permute_295);  view_552 = permute_295 = None
    add_tensor_37: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_37, arg412_1);  mm_default_37 = arg412_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_553: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_37, [1, 128, 1024]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_554: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_553, [1, -1, 16, 64]);  view_553 = None
    permute_296: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    clone_211: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_558: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_211, [16, -1, 64]);  clone_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_57: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_28, view_558);  div_28 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_561: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_57, [1, 16, 128, 64]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_299: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_214: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_562: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_214, [1, 128, 1024]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_563: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_562, [128, 1024]);  view_562 = None
    permute_300: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[128, 1024]" = torch.ops.aten.mm.default(view_563, permute_300);  view_563 = permute_300 = None
    add_tensor_36: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_36, arg414_1);  mm_default_36 = arg414_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_564: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_36, [1, 128, 1024]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_180: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_176, view_564);  add_176 = view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_50 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 128, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 128, 1]" = var_mean_50[1];  var_mean_50 = None
    sub_79: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_180, getitem_101);  getitem_101 = None
    add_181: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    mul_191: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_50);  sub_79 = rsqrt_50 = None
    mul_192: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_191, arg415_1);  mul_191 = arg415_1 = None
    add_182: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_192, arg416_1);  mul_192 = arg416_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_565: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_182, [128, 1024]);  add_182 = None
    permute_301: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[128, 1024]" = torch.ops.aten.mm.default(view_565, permute_301);  view_565 = permute_301 = None
    add_tensor_35: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_35, arg418_1);  mm_default_35 = arg418_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_566: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 128, 1024]);  add_tensor_35 = None
    mul_193: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_566, 0.125);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_573: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_193, [1, 128, 16, 64]);  mul_193 = None
    permute_306: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    clone_218: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_574: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_218, [16, -1, 64]);  clone_218 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_9: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_574, 0);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_567: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_302: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[128, 1024]" = torch.ops.aten.mm.default(view_567, permute_302);  view_567 = permute_302 = None
    add_tensor_34: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_34, arg420_1);  mm_default_34 = arg420_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_568: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 128, 1024]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_569: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_568, [1, -1, 16, 64]);  view_568 = None
    permute_303: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
    clone_216: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_575: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_216, [16, -1, 64]);  clone_216 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_10: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_575, 0);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_570: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_304: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[128, 1024]" = torch.ops.aten.mm.default(view_570, permute_304);  view_570 = permute_304 = None
    add_tensor_33: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_33, arg422_1);  mm_default_33 = arg422_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_571: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 128, 1024]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_572: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_571, [1, -1, 16, 64]);  view_571 = None
    permute_305: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
    clone_217: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_576: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_217, [16, -1, 64]);  clone_217 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_11: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_576, 0);  view_576 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, None, True, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
    getitem_127: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
    squeeze_dim_3: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_127, 0);  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_577: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_3, [1, 16, 128, 64]);  squeeze_dim_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_308: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_220: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    view_578: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_220, [1, 128, 1024]);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_579: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_578, [128, 1024]);  view_578 = None
    permute_309: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[128, 1024]" = torch.ops.aten.mm.default(view_579, permute_309);  view_579 = permute_309 = None
    add_tensor_32: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_32, arg424_1);  mm_default_32 = arg424_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_580: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 128, 1024]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_183: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_180, view_580);  add_180 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_51 = torch.ops.aten.var_mean.correction(add_183, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 128, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 128, 1]" = var_mean_51[1];  var_mean_51 = None
    sub_81: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_183, getitem_103);  getitem_103 = None
    add_184: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    mul_194: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_51);  sub_81 = rsqrt_51 = None
    mul_195: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_194, arg425_1);  mul_194 = arg425_1 = None
    add_185: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_195, arg426_1);  mul_195 = arg426_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_581: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_185, [128, 1024]);  add_185 = None
    permute_310: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[128, 4096]" = torch.ops.aten.mm.default(view_581, permute_310);  view_581 = permute_310 = None
    add_tensor_31: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_31, arg428_1);  mm_default_31 = arg428_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_582: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 128, 4096]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_196: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_582, 0.5)
    mul_197: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_582, 0.7071067811865476);  view_582 = None
    erf_20: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
    add_186: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_198: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_196, add_186);  mul_196 = add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_583: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_198, [128, 4096]);  mul_198 = None
    permute_311: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[128, 1024]" = torch.ops.aten.mm.default(view_583, permute_311);  view_583 = permute_311 = None
    add_tensor_30: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_30, arg430_1);  mm_default_30 = arg430_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_584: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 128, 1024]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_187: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_183, view_584);  add_183 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_52 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 128, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 128, 1]" = var_mean_52[1];  var_mean_52 = None
    sub_82: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_187, getitem_105);  getitem_105 = None
    add_188: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    mul_199: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_52);  sub_82 = rsqrt_52 = None
    mul_200: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_199, arg431_1);  mul_199 = arg431_1 = None
    add_189: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_200, arg432_1);  mul_200 = arg432_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_585: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_189, [128, 1024])
    permute_312: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[128, 1024]" = torch.ops.aten.mm.default(view_585, permute_312);  view_585 = permute_312 = None
    add_tensor_29: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_29, arg434_1);  mm_default_29 = arg434_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_586: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 128, 1024]);  add_tensor_29 = None
    mul_201: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_586, 0.125);  view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_593: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_201, [1, 128, 16, 64]);  mul_201 = None
    permute_317: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    clone_226: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_594: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_226, [16, -1, 64]);  clone_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_587: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_189, [128, 1024])
    permute_313: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[128, 1024]" = torch.ops.aten.mm.default(view_587, permute_313);  view_587 = permute_313 = None
    add_tensor_28: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_28, arg436_1);  mm_default_28 = arg436_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_588: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 128, 1024]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_589: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_588, [1, -1, 16, 64]);  view_588 = None
    permute_314: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
    clone_224: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_595: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_224, [16, -1, 64]);  clone_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_318: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_595, [0, 2, 1]);  view_595 = None
    bmm_60: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_594, permute_318);  view_594 = permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_597: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_60, [1, 16, 128, 128]);  bmm_60 = None
    add_190: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_597, expand_1);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_598: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_190, [16, 128, 128]);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_30: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_598, [-1], True)
    sub_83: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_598, amax_30);  view_598 = amax_30 = None
    exp_30: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
    sum_31: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
    div_30: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_590: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_189, [128, 1024]);  add_189 = None
    permute_315: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[128, 1024]" = torch.ops.aten.mm.default(view_590, permute_315);  view_590 = permute_315 = None
    add_tensor_27: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_27, arg438_1);  mm_default_27 = arg438_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_591: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 128, 1024]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_592: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_591, [1, -1, 16, 64]);  view_591 = None
    permute_316: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
    clone_225: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_596: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_225, [16, -1, 64]);  clone_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_61: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_30, view_596);  div_30 = view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_599: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_61, [1, 16, 128, 64]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_319: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_228: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_600: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_228, [1, 128, 1024]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_601: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_600, [128, 1024]);  view_600 = None
    permute_320: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[128, 1024]" = torch.ops.aten.mm.default(view_601, permute_320);  view_601 = permute_320 = None
    add_tensor_26: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_26, arg440_1);  mm_default_26 = arg440_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_602: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 128, 1024]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_191: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_187, view_602);  add_187 = view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_53 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 128, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 128, 1]" = var_mean_53[1];  var_mean_53 = None
    sub_84: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_191, getitem_107);  getitem_107 = None
    add_192: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    mul_202: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_53);  sub_84 = rsqrt_53 = None
    mul_203: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_202, arg441_1);  mul_202 = arg441_1 = None
    add_193: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_203, arg442_1);  mul_203 = arg442_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_603: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_193, [128, 1024]);  add_193 = None
    permute_321: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[128, 1024]" = torch.ops.aten.mm.default(view_603, permute_321);  view_603 = permute_321 = None
    add_tensor_25: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_25, arg444_1);  mm_default_25 = arg444_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_604: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 128, 1024]);  add_tensor_25 = None
    mul_204: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_604, 0.125);  view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_611: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_204, [1, 128, 16, 64]);  mul_204 = None
    permute_326: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None
    clone_232: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_612: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_232, [16, -1, 64]);  clone_232 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_6: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_612, 0);  view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_605: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_322: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[128, 1024]" = torch.ops.aten.mm.default(view_605, permute_322);  view_605 = permute_322 = None
    add_tensor_24: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_24, arg446_1);  mm_default_24 = arg446_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_606: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 128, 1024]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_607: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_606, [1, -1, 16, 64]);  view_606 = None
    permute_323: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
    clone_230: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_613: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_230, [16, -1, 64]);  clone_230 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_7: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_613, 0);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_608: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_324: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[128, 1024]" = torch.ops.aten.mm.default(view_608, permute_324);  view_608 = permute_324 = None
    add_tensor_23: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_23, arg448_1);  mm_default_23 = arg448_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_609: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 128, 1024]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_610: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_609, [1, -1, 16, 64]);  view_609 = None
    permute_325: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
    clone_231: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_614: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_231, [16, -1, 64]);  clone_231 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_8: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_614, 0);  view_614 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, None, True, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
    getitem_126: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
    squeeze_dim_2: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_126, 0);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_615: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_2, [1, 16, 128, 64]);  squeeze_dim_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_328: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_234: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_616: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_234, [1, 128, 1024]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_617: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_616, [128, 1024]);  view_616 = None
    permute_329: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[128, 1024]" = torch.ops.aten.mm.default(view_617, permute_329);  view_617 = permute_329 = None
    add_tensor_22: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_22, arg450_1);  mm_default_22 = arg450_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_618: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 128, 1024]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_194: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_191, view_618);  add_191 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_54 = torch.ops.aten.var_mean.correction(add_194, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 128, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 128, 1]" = var_mean_54[1];  var_mean_54 = None
    sub_86: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_194, getitem_109);  getitem_109 = None
    add_195: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    mul_205: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_54);  sub_86 = rsqrt_54 = None
    mul_206: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_205, arg451_1);  mul_205 = arg451_1 = None
    add_196: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_206, arg452_1);  mul_206 = arg452_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_619: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_196, [128, 1024]);  add_196 = None
    permute_330: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[128, 4096]" = torch.ops.aten.mm.default(view_619, permute_330);  view_619 = permute_330 = None
    add_tensor_21: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_21, arg454_1);  mm_default_21 = arg454_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_620: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 128, 4096]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_207: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_620, 0.5)
    mul_208: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_620, 0.7071067811865476);  view_620 = None
    erf_21: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_197: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_209: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_207, add_197);  mul_207 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_621: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_209, [128, 4096]);  mul_209 = None
    permute_331: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[128, 1024]" = torch.ops.aten.mm.default(view_621, permute_331);  view_621 = permute_331 = None
    add_tensor_20: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_20, arg456_1);  mm_default_20 = arg456_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_622: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 128, 1024]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_198: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_194, view_622);  add_194 = view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_55 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 128, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 128, 1]" = var_mean_55[1];  var_mean_55 = None
    sub_87: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_198, getitem_111);  getitem_111 = None
    add_199: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    mul_210: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_55);  sub_87 = rsqrt_55 = None
    mul_211: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_210, arg457_1);  mul_210 = arg457_1 = None
    add_200: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_211, arg458_1);  mul_211 = arg458_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_623: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_200, [128, 1024])
    permute_332: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[128, 1024]" = torch.ops.aten.mm.default(view_623, permute_332);  view_623 = permute_332 = None
    add_tensor_19: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_19, arg460_1);  mm_default_19 = arg460_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_624: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 128, 1024]);  add_tensor_19 = None
    mul_212: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_624, 0.125);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_631: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_212, [1, 128, 16, 64]);  mul_212 = None
    permute_337: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
    clone_240: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_632: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_240, [16, -1, 64]);  clone_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_625: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_200, [128, 1024])
    permute_333: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[128, 1024]" = torch.ops.aten.mm.default(view_625, permute_333);  view_625 = permute_333 = None
    add_tensor_18: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_18, arg462_1);  mm_default_18 = arg462_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_626: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 128, 1024]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_627: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_626, [1, -1, 16, 64]);  view_626 = None
    permute_334: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
    clone_238: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_633: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_238, [16, -1, 64]);  clone_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_338: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_633, [0, 2, 1]);  view_633 = None
    bmm_64: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_632, permute_338);  view_632 = permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_635: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_64, [1, 16, 128, 128]);  bmm_64 = None
    add_201: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_635, expand_1);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_636: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_201, [16, 128, 128]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_32: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_636, [-1], True)
    sub_88: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_636, amax_32);  view_636 = amax_32 = None
    exp_32: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_88);  sub_88 = None
    sum_33: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
    div_32: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_628: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_200, [128, 1024]);  add_200 = None
    permute_335: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[128, 1024]" = torch.ops.aten.mm.default(view_628, permute_335);  view_628 = permute_335 = None
    add_tensor_17: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_17, arg464_1);  mm_default_17 = arg464_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_629: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 128, 1024]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_630: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_629, [1, -1, 16, 64]);  view_629 = None
    permute_336: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    clone_239: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_634: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_239, [16, -1, 64]);  clone_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_65: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_32, view_634);  div_32 = view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_637: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_65, [1, 16, 128, 64]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_339: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_637, [0, 2, 1, 3]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_242: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_339, memory_format = torch.contiguous_format);  permute_339 = None
    view_638: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_242, [1, 128, 1024]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_639: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_638, [128, 1024]);  view_638 = None
    permute_340: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[128, 1024]" = torch.ops.aten.mm.default(view_639, permute_340);  view_639 = permute_340 = None
    add_tensor_16: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_16, arg466_1);  mm_default_16 = arg466_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_640: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 128, 1024]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_202: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_198, view_640);  add_198 = view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_56 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 128, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 128, 1]" = var_mean_56[1];  var_mean_56 = None
    sub_89: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_202, getitem_113);  getitem_113 = None
    add_203: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    mul_213: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_56);  sub_89 = rsqrt_56 = None
    mul_214: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_213, arg467_1);  mul_213 = arg467_1 = None
    add_204: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_214, arg468_1);  mul_214 = arg468_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_641: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_204, [128, 1024]);  add_204 = None
    permute_341: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[128, 1024]" = torch.ops.aten.mm.default(view_641, permute_341);  view_641 = permute_341 = None
    add_tensor_15: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_15, arg470_1);  mm_default_15 = arg470_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_642: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 128, 1024]);  add_tensor_15 = None
    mul_215: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_642, 0.125);  view_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_649: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_215, [1, 128, 16, 64]);  mul_215 = None
    permute_346: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
    clone_246: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_650: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_246, [16, -1, 64]);  clone_246 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_3: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_650, 0);  view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_643: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_342: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[128, 1024]" = torch.ops.aten.mm.default(view_643, permute_342);  view_643 = permute_342 = None
    add_tensor_14: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_14, arg472_1);  mm_default_14 = arg472_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_644: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 128, 1024]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_645: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_644, [1, -1, 16, 64]);  view_644 = None
    permute_343: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
    clone_244: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_651: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_244, [16, -1, 64]);  clone_244 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_4: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_651, 0);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_646: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_344: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[128, 1024]" = torch.ops.aten.mm.default(view_646, permute_344);  view_646 = permute_344 = None
    add_tensor_13: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_13, arg474_1);  mm_default_13 = arg474_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_647: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 128, 1024]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_648: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_647, [1, -1, 16, 64]);  view_647 = None
    permute_345: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
    clone_245: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_652: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_245, [16, -1, 64]);  clone_245 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_5: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_652, 0);  view_652 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, None, True, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
    getitem_125: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
    squeeze_dim_1: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_125, 0);  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_653: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim_1, [1, 16, 128, 64]);  squeeze_dim_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_348: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_248: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_654: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_248, [1, 128, 1024]);  clone_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_655: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_654, [128, 1024]);  view_654 = None
    permute_349: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[128, 1024]" = torch.ops.aten.mm.default(view_655, permute_349);  view_655 = permute_349 = None
    add_tensor_12: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_12, arg476_1);  mm_default_12 = arg476_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_656: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 128, 1024]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_205: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_202, view_656);  add_202 = view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_57 = torch.ops.aten.var_mean.correction(add_205, [2], correction = 0, keepdim = True)
    getitem_114: "f32[1, 128, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 128, 1]" = var_mean_57[1];  var_mean_57 = None
    sub_91: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_205, getitem_115);  getitem_115 = None
    add_206: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_57: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    mul_216: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_57);  sub_91 = rsqrt_57 = None
    mul_217: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_216, arg477_1);  mul_216 = arg477_1 = None
    add_207: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_217, arg478_1);  mul_217 = arg478_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_657: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_207, [128, 1024]);  add_207 = None
    permute_350: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[128, 4096]" = torch.ops.aten.mm.default(view_657, permute_350);  view_657 = permute_350 = None
    add_tensor_11: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_11, arg480_1);  mm_default_11 = arg480_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_658: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 128, 4096]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_218: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_658, 0.5)
    mul_219: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_658, 0.7071067811865476);  view_658 = None
    erf_22: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_208: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_220: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_218, add_208);  mul_218 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_659: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_220, [128, 4096]);  mul_220 = None
    permute_351: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[128, 1024]" = torch.ops.aten.mm.default(view_659, permute_351);  view_659 = permute_351 = None
    add_tensor_10: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_10, arg482_1);  mm_default_10 = arg482_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_660: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 128, 1024]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_209: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_205, view_660);  add_205 = view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_58 = torch.ops.aten.var_mean.correction(add_209, [2], correction = 0, keepdim = True)
    getitem_116: "f32[1, 128, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 128, 1]" = var_mean_58[1];  var_mean_58 = None
    sub_92: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_209, getitem_117);  getitem_117 = None
    add_210: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_58: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    mul_221: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_58);  sub_92 = rsqrt_58 = None
    mul_222: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_221, arg483_1);  mul_221 = arg483_1 = None
    add_211: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_222, arg484_1);  mul_222 = arg484_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_661: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_211, [128, 1024])
    permute_352: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg485_1, [1, 0]);  arg485_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[128, 1024]" = torch.ops.aten.mm.default(view_661, permute_352);  view_661 = permute_352 = None
    add_tensor_9: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_9, arg486_1);  mm_default_9 = arg486_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_662: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 128, 1024]);  add_tensor_9 = None
    mul_223: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_662, 0.125);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_669: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_223, [1, 128, 16, 64]);  mul_223 = None
    permute_357: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_669, [0, 2, 1, 3]);  view_669 = None
    clone_254: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_670: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_254, [16, -1, 64]);  clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_663: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_211, [128, 1024])
    permute_353: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[128, 1024]" = torch.ops.aten.mm.default(view_663, permute_353);  view_663 = permute_353 = None
    add_tensor_8: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_8, arg488_1);  mm_default_8 = arg488_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_664: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 128, 1024]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_665: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_664, [1, -1, 16, 64]);  view_664 = None
    permute_354: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_665, [0, 2, 1, 3]);  view_665 = None
    clone_252: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_671: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_252, [16, -1, 64]);  clone_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_358: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_671, [0, 2, 1]);  view_671 = None
    bmm_68: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_670, permute_358);  view_670 = permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_673: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_68, [1, 16, 128, 128]);  bmm_68 = None
    add_212: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_673, expand_1);  view_673 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_674: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_212, [16, 128, 128]);  add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_34: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_674, [-1], True)
    sub_93: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_674, amax_34);  view_674 = amax_34 = None
    exp_34: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
    sum_35: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
    div_34: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_666: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_211, [128, 1024]);  add_211 = None
    permute_355: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[128, 1024]" = torch.ops.aten.mm.default(view_666, permute_355);  view_666 = permute_355 = None
    add_tensor_7: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_7, arg490_1);  mm_default_7 = arg490_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_667: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 128, 1024]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_668: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_667, [1, -1, 16, 64]);  view_667 = None
    permute_356: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_668, [0, 2, 1, 3]);  view_668 = None
    clone_253: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_672: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_253, [16, -1, 64]);  clone_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_69: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_34, view_672);  div_34 = view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_675: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_69, [1, 16, 128, 64]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_359: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_675, [0, 2, 1, 3]);  view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_256: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_676: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_256, [1, 128, 1024]);  clone_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_677: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_676, [128, 1024]);  view_676 = None
    permute_360: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[128, 1024]" = torch.ops.aten.mm.default(view_677, permute_360);  view_677 = permute_360 = None
    add_tensor_6: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_6, arg492_1);  mm_default_6 = arg492_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_678: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 128, 1024]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_213: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_209, view_678);  add_209 = view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_59 = torch.ops.aten.var_mean.correction(add_213, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 128, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 128, 1]" = var_mean_59[1];  var_mean_59 = None
    sub_94: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_213, getitem_119);  getitem_119 = None
    add_214: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
    mul_224: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_59);  sub_94 = rsqrt_59 = None
    mul_225: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_224, arg493_1);  mul_224 = arg493_1 = None
    add_215: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_225, arg494_1);  mul_225 = arg494_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_679: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_215, [128, 1024]);  add_215 = None
    permute_361: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[128, 1024]" = torch.ops.aten.mm.default(view_679, permute_361);  view_679 = permute_361 = None
    add_tensor_5: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_5, arg496_1);  mm_default_5 = arg496_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_680: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 128, 1024]);  add_tensor_5 = None
    mul_226: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_680, 0.125);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_687: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_226, [1, 128, 16, 64]);  mul_226 = None
    permute_366: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_687, [0, 2, 1, 3]);  view_687 = None
    clone_260: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_688: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_260, [16, -1, 64]);  clone_260 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_688, 0);  view_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_681: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_362: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[128, 1024]" = torch.ops.aten.mm.default(view_681, permute_362);  view_681 = permute_362 = None
    add_tensor_4: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_4, arg498_1);  mm_default_4 = arg498_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_682: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 128, 1024]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_683: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_682, [1, -1, 16, 64]);  view_682 = None
    permute_363: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_683, [0, 2, 1, 3]);  view_683 = None
    clone_258: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_689: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_258, [16, -1, 64]);  clone_258 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_1: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_689, 0);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_684: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024])
    permute_364: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[128, 1024]" = torch.ops.aten.mm.default(view_684, permute_364);  view_684 = permute_364 = None
    add_tensor_3: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_3, arg500_1);  mm_default_3 = arg500_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_685: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 128, 1024]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_686: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_685, [1, -1, 16, 64]);  view_685 = None
    permute_365: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
    clone_259: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_690: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_259, [16, -1, 64]);  clone_259 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_2: "f32[1, 16, 128, 64]" = torch.ops.aten.unsqueeze.default(view_690, 0);  view_690 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, True, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem_124: "f32[1, 16, 128, 64]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    squeeze_dim: "f32[16, 128, 64]" = torch.ops.aten.squeeze.dim(getitem_124, 0);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_691: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(squeeze_dim, [1, 16, 128, 64]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_368: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_262: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_692: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_262, [1, 128, 1024]);  clone_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_693: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_692, [128, 1024]);  view_692 = None
    permute_369: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[128, 1024]" = torch.ops.aten.mm.default(view_693, permute_369);  view_693 = permute_369 = None
    add_tensor_2: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default_2, arg502_1);  mm_default_2 = arg502_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_694: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 128, 1024]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_216: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_213, view_694);  add_213 = view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_60 = torch.ops.aten.var_mean.correction(add_216, [2], correction = 0, keepdim = True)
    getitem_120: "f32[1, 128, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 128, 1]" = var_mean_60[1];  var_mean_60 = None
    sub_96: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_216, getitem_121);  getitem_121 = None
    add_217: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_60: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    mul_227: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_60);  sub_96 = rsqrt_60 = None
    mul_228: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_227, arg503_1);  mul_227 = arg503_1 = None
    add_218: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_228, arg504_1);  mul_228 = arg504_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_695: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_218, [128, 1024]);  add_218 = None
    permute_370: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[128, 4096]" = torch.ops.aten.mm.default(view_695, permute_370);  view_695 = permute_370 = None
    add_tensor_1: "f32[128, 4096]" = torch.ops.aten.add.Tensor(mm_default_1, arg506_1);  mm_default_1 = arg506_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_696: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 128, 4096]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_229: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_696, 0.5)
    mul_230: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_696, 0.7071067811865476);  view_696 = None
    erf_23: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_230);  mul_230 = None
    add_219: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_231: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_229, add_219);  mul_229 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_697: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_231, [128, 4096]);  mul_231 = None
    permute_371: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[128, 1024]" = torch.ops.aten.mm.default(view_697, permute_371);  view_697 = permute_371 = None
    add_tensor: "f32[128, 1024]" = torch.ops.aten.add.Tensor(mm_default, arg508_1);  mm_default = arg508_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_698: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor, [1, 128, 1024]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_220: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_216, view_698);  add_216 = view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1133, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_61 = torch.ops.aten.var_mean.correction(add_220, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 128, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 128, 1]" = var_mean_61[1];  var_mean_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1437, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_702: "i64[128]" = torch.ops.aten.reshape.default(arg513_1, [-1]);  arg513_1 = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_702, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1133, code: hidden_states = self.layer_norm(hidden_states)
    sub_97: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_220, getitem_123);  add_220 = getitem_123 = None
    add_221: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_61: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    mul_232: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_61);  sub_97 = rsqrt_61 = None
    mul_233: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_232, arg509_1);  mul_232 = arg509_1 = None
    add_222: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_233, arg510_1);  mul_233 = arg510_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1432, code: lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    view_699: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_222, [128, 1024]);  add_222 = None
    permute_372: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
    mm: "f32[128, 50265]" = torch.ops.aten.mm.default(view_699, permute_372);  view_699 = permute_372 = None
    view_700: "f32[1, 128, 50265]" = torch.ops.aten.reshape.default(mm, [1, 128, 50265]);  mm = None
    add_223: "f32[1, 128, 50265]" = torch.ops.aten.add.Tensor(view_700, arg512_1);  view_700 = arg512_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1437, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_701: "f32[128, 50265]" = torch.ops.aten.reshape.default(add_223, [-1, 50265])
    amax_36: "f32[128, 1]" = torch.ops.aten.amax.default(view_701, [1], True)
    sub_98: "f32[128, 50265]" = torch.ops.aten.sub.Tensor(view_701, amax_36);  view_701 = amax_36 = None
    exp_36: "f32[128, 50265]" = torch.ops.aten.exp.default(sub_98)
    sum_37: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [1], True);  exp_36 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_37);  sum_37 = None
    sub_99: "f32[128, 50265]" = torch.ops.aten.sub.Tensor(sub_98, log);  sub_98 = log = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_702, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "i64[128]" = torch.ops.aten.where.self(ne, view_702, full_default_2);  ne = full_default_2 = None
    unsqueeze_4: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_99, 1, unsqueeze_4);  sub_99 = unsqueeze_4 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_2: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
    sum_39: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_702, -100);  view_702 = None
    sum_38: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
    div_36: "f32[]" = torch.ops.aten.div.Tensor(sum_39, convert_element_type);  sum_39 = convert_element_type = None
    return (div_36, add_223, add_86)
    