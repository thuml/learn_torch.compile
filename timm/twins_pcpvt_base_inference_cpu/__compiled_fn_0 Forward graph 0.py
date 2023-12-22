from __future__ import annotations



def forward(self, arg0_1: "f32[64, 3, 4, 4]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64, 64]", arg7_1: "f32[64]", arg8_1: "f32[64, 64, 8, 8]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[128, 64]", arg13_1: "f32[128]", arg14_1: "f32[64, 64]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[512, 64]", arg19_1: "f32[512]", arg20_1: "f32[64, 512]", arg21_1: "f32[64]", arg22_1: "f32[64, 1, 3, 3]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[64, 64]", arg27_1: "f32[64]", arg28_1: "f32[64, 64, 8, 8]", arg29_1: "f32[64]", arg30_1: "f32[64]", arg31_1: "f32[64]", arg32_1: "f32[128, 64]", arg33_1: "f32[128]", arg34_1: "f32[64, 64]", arg35_1: "f32[64]", arg36_1: "f32[64]", arg37_1: "f32[64]", arg38_1: "f32[512, 64]", arg39_1: "f32[512]", arg40_1: "f32[64, 512]", arg41_1: "f32[64]", arg42_1: "f32[64]", arg43_1: "f32[64]", arg44_1: "f32[64, 64]", arg45_1: "f32[64]", arg46_1: "f32[64, 64, 8, 8]", arg47_1: "f32[64]", arg48_1: "f32[64]", arg49_1: "f32[64]", arg50_1: "f32[128, 64]", arg51_1: "f32[128]", arg52_1: "f32[64, 64]", arg53_1: "f32[64]", arg54_1: "f32[64]", arg55_1: "f32[64]", arg56_1: "f32[512, 64]", arg57_1: "f32[512]", arg58_1: "f32[64, 512]", arg59_1: "f32[64]", arg60_1: "f32[128, 64, 2, 2]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[128]", arg66_1: "f32[128, 128]", arg67_1: "f32[128]", arg68_1: "f32[128, 128, 4, 4]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[128]", arg72_1: "f32[256, 128]", arg73_1: "f32[256]", arg74_1: "f32[128, 128]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[1024, 128]", arg79_1: "f32[1024]", arg80_1: "f32[128, 1024]", arg81_1: "f32[128]", arg82_1: "f32[128, 1, 3, 3]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128, 128]", arg87_1: "f32[128]", arg88_1: "f32[128, 128, 4, 4]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[128]", arg92_1: "f32[256, 128]", arg93_1: "f32[256]", arg94_1: "f32[128, 128]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "f32[128]", arg98_1: "f32[1024, 128]", arg99_1: "f32[1024]", arg100_1: "f32[128, 1024]", arg101_1: "f32[128]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128, 128]", arg105_1: "f32[128]", arg106_1: "f32[128, 128, 4, 4]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[256, 128]", arg111_1: "f32[256]", arg112_1: "f32[128, 128]", arg113_1: "f32[128]", arg114_1: "f32[128]", arg115_1: "f32[128]", arg116_1: "f32[1024, 128]", arg117_1: "f32[1024]", arg118_1: "f32[128, 1024]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[128]", arg122_1: "f32[128, 128]", arg123_1: "f32[128]", arg124_1: "f32[128, 128, 4, 4]", arg125_1: "f32[128]", arg126_1: "f32[128]", arg127_1: "f32[128]", arg128_1: "f32[256, 128]", arg129_1: "f32[256]", arg130_1: "f32[128, 128]", arg131_1: "f32[128]", arg132_1: "f32[128]", arg133_1: "f32[128]", arg134_1: "f32[1024, 128]", arg135_1: "f32[1024]", arg136_1: "f32[128, 1024]", arg137_1: "f32[128]", arg138_1: "f32[320, 128, 2, 2]", arg139_1: "f32[320]", arg140_1: "f32[320]", arg141_1: "f32[320]", arg142_1: "f32[320]", arg143_1: "f32[320]", arg144_1: "f32[320, 320]", arg145_1: "f32[320]", arg146_1: "f32[320, 320, 2, 2]", arg147_1: "f32[320]", arg148_1: "f32[320]", arg149_1: "f32[320]", arg150_1: "f32[640, 320]", arg151_1: "f32[640]", arg152_1: "f32[320, 320]", arg153_1: "f32[320]", arg154_1: "f32[320]", arg155_1: "f32[320]", arg156_1: "f32[1280, 320]", arg157_1: "f32[1280]", arg158_1: "f32[320, 1280]", arg159_1: "f32[320]", arg160_1: "f32[320, 1, 3, 3]", arg161_1: "f32[320]", arg162_1: "f32[320]", arg163_1: "f32[320]", arg164_1: "f32[320, 320]", arg165_1: "f32[320]", arg166_1: "f32[320, 320, 2, 2]", arg167_1: "f32[320]", arg168_1: "f32[320]", arg169_1: "f32[320]", arg170_1: "f32[640, 320]", arg171_1: "f32[640]", arg172_1: "f32[320, 320]", arg173_1: "f32[320]", arg174_1: "f32[320]", arg175_1: "f32[320]", arg176_1: "f32[1280, 320]", arg177_1: "f32[1280]", arg178_1: "f32[320, 1280]", arg179_1: "f32[320]", arg180_1: "f32[320]", arg181_1: "f32[320]", arg182_1: "f32[320, 320]", arg183_1: "f32[320]", arg184_1: "f32[320, 320, 2, 2]", arg185_1: "f32[320]", arg186_1: "f32[320]", arg187_1: "f32[320]", arg188_1: "f32[640, 320]", arg189_1: "f32[640]", arg190_1: "f32[320, 320]", arg191_1: "f32[320]", arg192_1: "f32[320]", arg193_1: "f32[320]", arg194_1: "f32[1280, 320]", arg195_1: "f32[1280]", arg196_1: "f32[320, 1280]", arg197_1: "f32[320]", arg198_1: "f32[320]", arg199_1: "f32[320]", arg200_1: "f32[320, 320]", arg201_1: "f32[320]", arg202_1: "f32[320, 320, 2, 2]", arg203_1: "f32[320]", arg204_1: "f32[320]", arg205_1: "f32[320]", arg206_1: "f32[640, 320]", arg207_1: "f32[640]", arg208_1: "f32[320, 320]", arg209_1: "f32[320]", arg210_1: "f32[320]", arg211_1: "f32[320]", arg212_1: "f32[1280, 320]", arg213_1: "f32[1280]", arg214_1: "f32[320, 1280]", arg215_1: "f32[320]", arg216_1: "f32[320]", arg217_1: "f32[320]", arg218_1: "f32[320, 320]", arg219_1: "f32[320]", arg220_1: "f32[320, 320, 2, 2]", arg221_1: "f32[320]", arg222_1: "f32[320]", arg223_1: "f32[320]", arg224_1: "f32[640, 320]", arg225_1: "f32[640]", arg226_1: "f32[320, 320]", arg227_1: "f32[320]", arg228_1: "f32[320]", arg229_1: "f32[320]", arg230_1: "f32[1280, 320]", arg231_1: "f32[1280]", arg232_1: "f32[320, 1280]", arg233_1: "f32[320]", arg234_1: "f32[320]", arg235_1: "f32[320]", arg236_1: "f32[320, 320]", arg237_1: "f32[320]", arg238_1: "f32[320, 320, 2, 2]", arg239_1: "f32[320]", arg240_1: "f32[320]", arg241_1: "f32[320]", arg242_1: "f32[640, 320]", arg243_1: "f32[640]", arg244_1: "f32[320, 320]", arg245_1: "f32[320]", arg246_1: "f32[320]", arg247_1: "f32[320]", arg248_1: "f32[1280, 320]", arg249_1: "f32[1280]", arg250_1: "f32[320, 1280]", arg251_1: "f32[320]", arg252_1: "f32[320]", arg253_1: "f32[320]", arg254_1: "f32[320, 320]", arg255_1: "f32[320]", arg256_1: "f32[320, 320, 2, 2]", arg257_1: "f32[320]", arg258_1: "f32[320]", arg259_1: "f32[320]", arg260_1: "f32[640, 320]", arg261_1: "f32[640]", arg262_1: "f32[320, 320]", arg263_1: "f32[320]", arg264_1: "f32[320]", arg265_1: "f32[320]", arg266_1: "f32[1280, 320]", arg267_1: "f32[1280]", arg268_1: "f32[320, 1280]", arg269_1: "f32[320]", arg270_1: "f32[320]", arg271_1: "f32[320]", arg272_1: "f32[320, 320]", arg273_1: "f32[320]", arg274_1: "f32[320, 320, 2, 2]", arg275_1: "f32[320]", arg276_1: "f32[320]", arg277_1: "f32[320]", arg278_1: "f32[640, 320]", arg279_1: "f32[640]", arg280_1: "f32[320, 320]", arg281_1: "f32[320]", arg282_1: "f32[320]", arg283_1: "f32[320]", arg284_1: "f32[1280, 320]", arg285_1: "f32[1280]", arg286_1: "f32[320, 1280]", arg287_1: "f32[320]", arg288_1: "f32[320]", arg289_1: "f32[320]", arg290_1: "f32[320, 320]", arg291_1: "f32[320]", arg292_1: "f32[320, 320, 2, 2]", arg293_1: "f32[320]", arg294_1: "f32[320]", arg295_1: "f32[320]", arg296_1: "f32[640, 320]", arg297_1: "f32[640]", arg298_1: "f32[320, 320]", arg299_1: "f32[320]", arg300_1: "f32[320]", arg301_1: "f32[320]", arg302_1: "f32[1280, 320]", arg303_1: "f32[1280]", arg304_1: "f32[320, 1280]", arg305_1: "f32[320]", arg306_1: "f32[320]", arg307_1: "f32[320]", arg308_1: "f32[320, 320]", arg309_1: "f32[320]", arg310_1: "f32[320, 320, 2, 2]", arg311_1: "f32[320]", arg312_1: "f32[320]", arg313_1: "f32[320]", arg314_1: "f32[640, 320]", arg315_1: "f32[640]", arg316_1: "f32[320, 320]", arg317_1: "f32[320]", arg318_1: "f32[320]", arg319_1: "f32[320]", arg320_1: "f32[1280, 320]", arg321_1: "f32[1280]", arg322_1: "f32[320, 1280]", arg323_1: "f32[320]", arg324_1: "f32[320]", arg325_1: "f32[320]", arg326_1: "f32[320, 320]", arg327_1: "f32[320]", arg328_1: "f32[320, 320, 2, 2]", arg329_1: "f32[320]", arg330_1: "f32[320]", arg331_1: "f32[320]", arg332_1: "f32[640, 320]", arg333_1: "f32[640]", arg334_1: "f32[320, 320]", arg335_1: "f32[320]", arg336_1: "f32[320]", arg337_1: "f32[320]", arg338_1: "f32[1280, 320]", arg339_1: "f32[1280]", arg340_1: "f32[320, 1280]", arg341_1: "f32[320]", arg342_1: "f32[320]", arg343_1: "f32[320]", arg344_1: "f32[320, 320]", arg345_1: "f32[320]", arg346_1: "f32[320, 320, 2, 2]", arg347_1: "f32[320]", arg348_1: "f32[320]", arg349_1: "f32[320]", arg350_1: "f32[640, 320]", arg351_1: "f32[640]", arg352_1: "f32[320, 320]", arg353_1: "f32[320]", arg354_1: "f32[320]", arg355_1: "f32[320]", arg356_1: "f32[1280, 320]", arg357_1: "f32[1280]", arg358_1: "f32[320, 1280]", arg359_1: "f32[320]", arg360_1: "f32[320]", arg361_1: "f32[320]", arg362_1: "f32[320, 320]", arg363_1: "f32[320]", arg364_1: "f32[320, 320, 2, 2]", arg365_1: "f32[320]", arg366_1: "f32[320]", arg367_1: "f32[320]", arg368_1: "f32[640, 320]", arg369_1: "f32[640]", arg370_1: "f32[320, 320]", arg371_1: "f32[320]", arg372_1: "f32[320]", arg373_1: "f32[320]", arg374_1: "f32[1280, 320]", arg375_1: "f32[1280]", arg376_1: "f32[320, 1280]", arg377_1: "f32[320]", arg378_1: "f32[320]", arg379_1: "f32[320]", arg380_1: "f32[320, 320]", arg381_1: "f32[320]", arg382_1: "f32[320, 320, 2, 2]", arg383_1: "f32[320]", arg384_1: "f32[320]", arg385_1: "f32[320]", arg386_1: "f32[640, 320]", arg387_1: "f32[640]", arg388_1: "f32[320, 320]", arg389_1: "f32[320]", arg390_1: "f32[320]", arg391_1: "f32[320]", arg392_1: "f32[1280, 320]", arg393_1: "f32[1280]", arg394_1: "f32[320, 1280]", arg395_1: "f32[320]", arg396_1: "f32[320]", arg397_1: "f32[320]", arg398_1: "f32[320, 320]", arg399_1: "f32[320]", arg400_1: "f32[320, 320, 2, 2]", arg401_1: "f32[320]", arg402_1: "f32[320]", arg403_1: "f32[320]", arg404_1: "f32[640, 320]", arg405_1: "f32[640]", arg406_1: "f32[320, 320]", arg407_1: "f32[320]", arg408_1: "f32[320]", arg409_1: "f32[320]", arg410_1: "f32[1280, 320]", arg411_1: "f32[1280]", arg412_1: "f32[320, 1280]", arg413_1: "f32[320]", arg414_1: "f32[320]", arg415_1: "f32[320]", arg416_1: "f32[320, 320]", arg417_1: "f32[320]", arg418_1: "f32[320, 320, 2, 2]", arg419_1: "f32[320]", arg420_1: "f32[320]", arg421_1: "f32[320]", arg422_1: "f32[640, 320]", arg423_1: "f32[640]", arg424_1: "f32[320, 320]", arg425_1: "f32[320]", arg426_1: "f32[320]", arg427_1: "f32[320]", arg428_1: "f32[1280, 320]", arg429_1: "f32[1280]", arg430_1: "f32[320, 1280]", arg431_1: "f32[320]", arg432_1: "f32[320]", arg433_1: "f32[320]", arg434_1: "f32[320, 320]", arg435_1: "f32[320]", arg436_1: "f32[320, 320, 2, 2]", arg437_1: "f32[320]", arg438_1: "f32[320]", arg439_1: "f32[320]", arg440_1: "f32[640, 320]", arg441_1: "f32[640]", arg442_1: "f32[320, 320]", arg443_1: "f32[320]", arg444_1: "f32[320]", arg445_1: "f32[320]", arg446_1: "f32[1280, 320]", arg447_1: "f32[1280]", arg448_1: "f32[320, 1280]", arg449_1: "f32[320]", arg450_1: "f32[320]", arg451_1: "f32[320]", arg452_1: "f32[320, 320]", arg453_1: "f32[320]", arg454_1: "f32[320, 320, 2, 2]", arg455_1: "f32[320]", arg456_1: "f32[320]", arg457_1: "f32[320]", arg458_1: "f32[640, 320]", arg459_1: "f32[640]", arg460_1: "f32[320, 320]", arg461_1: "f32[320]", arg462_1: "f32[320]", arg463_1: "f32[320]", arg464_1: "f32[1280, 320]", arg465_1: "f32[1280]", arg466_1: "f32[320, 1280]", arg467_1: "f32[320]", arg468_1: "f32[512, 320, 2, 2]", arg469_1: "f32[512]", arg470_1: "f32[512]", arg471_1: "f32[512]", arg472_1: "f32[512]", arg473_1: "f32[512]", arg474_1: "f32[512, 512]", arg475_1: "f32[512]", arg476_1: "f32[1024, 512]", arg477_1: "f32[1024]", arg478_1: "f32[512, 512]", arg479_1: "f32[512]", arg480_1: "f32[512]", arg481_1: "f32[512]", arg482_1: "f32[2048, 512]", arg483_1: "f32[2048]", arg484_1: "f32[512, 2048]", arg485_1: "f32[512]", arg486_1: "f32[512, 1, 3, 3]", arg487_1: "f32[512]", arg488_1: "f32[512]", arg489_1: "f32[512]", arg490_1: "f32[512, 512]", arg491_1: "f32[512]", arg492_1: "f32[1024, 512]", arg493_1: "f32[1024]", arg494_1: "f32[512, 512]", arg495_1: "f32[512]", arg496_1: "f32[512]", arg497_1: "f32[512]", arg498_1: "f32[2048, 512]", arg499_1: "f32[2048]", arg500_1: "f32[512, 2048]", arg501_1: "f32[512]", arg502_1: "f32[512]", arg503_1: "f32[512]", arg504_1: "f32[512, 512]", arg505_1: "f32[512]", arg506_1: "f32[1024, 512]", arg507_1: "f32[1024]", arg508_1: "f32[512, 512]", arg509_1: "f32[512]", arg510_1: "f32[512]", arg511_1: "f32[512]", arg512_1: "f32[2048, 512]", arg513_1: "f32[2048]", arg514_1: "f32[512, 2048]", arg515_1: "f32[512]", arg516_1: "f32[512]", arg517_1: "f32[512]", arg518_1: "f32[1000, 512]", arg519_1: "f32[1000]", arg520_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(arg520_1, arg0_1, arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg520_1 = arg0_1 = arg1_1 = None
    view: "f32[8, 64, 3136]" = torch.ops.aten.view.default(convolution, [8, 64, 3136]);  convolution = None
    permute: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 3136, 1]" = var_mean[0]
    getitem_1: "f32[8, 3136, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_1: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_1: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 3136, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 3136, 1]" = var_mean_1[1];  var_mean_1 = None
    add_2: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_1, getitem_3);  getitem_3 = None
    mul_2: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_3: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    add_3: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_1: "f32[25088, 64]" = torch.ops.aten.view.default(add_3, [25088, 64])
    permute_1: "f32[64, 64]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg7_1, view_1, permute_1);  arg7_1 = view_1 = permute_1 = None
    view_2: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm, [8, 3136, 64]);  addmm = None
    view_3: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_2, [8, 3136, 1, 64]);  view_2 = None
    permute_2: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_3: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_3, [0, 2, 1]);  add_3 = None
    view_4: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_3, [8, 64, 56, 56]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_1: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_4, arg8_1, arg9_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_4 = arg8_1 = arg9_1 = None
    view_5: "f32[8, 64, 49]" = torch.ops.aten.view.default(convolution_1, [8, 64, 49]);  convolution_1 = None
    permute_4: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_2 = torch.ops.aten.var_mean.correction(permute_4, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 49, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 49, 1]" = var_mean_2[1];  var_mean_2 = None
    add_4: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(permute_4, getitem_5);  permute_4 = getitem_5 = None
    mul_4: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_5: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_4, arg10_1);  mul_4 = arg10_1 = None
    add_5: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_5, arg11_1);  mul_5 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_6: "f32[392, 64]" = torch.ops.aten.view.default(add_5, [392, 64]);  add_5 = None
    permute_5: "f32[64, 128]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    addmm_1: "f32[392, 128]" = torch.ops.aten.addmm.default(arg13_1, view_6, permute_5);  arg13_1 = view_6 = permute_5 = None
    view_7: "f32[8, 49, 128]" = torch.ops.aten.view.default(addmm_1, [8, 49, 128]);  addmm_1 = None
    view_8: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.view.default(view_7, [8, -1, 2, 1, 64]);  view_7 = None
    permute_6: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_8, [2, 0, 3, 1, 4]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_6);  permute_6 = None
    getitem_6: "f32[8, 1, 49, 64]" = unbind[0]
    getitem_7: "f32[8, 1, 49, 64]" = unbind[1];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_2, getitem_6, getitem_7);  permute_2 = getitem_6 = getitem_7 = None
    getitem_8: "f32[8, 1, 3136, 64]" = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_7: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
    view_9: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_7, [8, 3136, 64]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_10: "f32[25088, 64]" = torch.ops.aten.view.default(view_9, [25088, 64]);  view_9 = None
    permute_8: "f32[64, 64]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_2: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg15_1, view_10, permute_8);  arg15_1 = view_10 = permute_8 = None
    view_11: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_2, [8, 3136, 64]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_2: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_11);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_6: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(clone_1, clone_2);  clone_1 = clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_17: "f32[8, 3136, 1]" = var_mean_3[0]
    getitem_18: "f32[8, 3136, 1]" = var_mean_3[1];  var_mean_3 = None
    add_7: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_17, 1e-06);  getitem_17 = None
    rsqrt_3: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_3: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_6, getitem_18);  getitem_18 = None
    mul_6: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_7: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_6, arg16_1);  mul_6 = arg16_1 = None
    add_8: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_7, arg17_1);  mul_7 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_12: "f32[25088, 64]" = torch.ops.aten.view.default(add_8, [25088, 64]);  add_8 = None
    permute_9: "f32[64, 512]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    addmm_3: "f32[25088, 512]" = torch.ops.aten.addmm.default(arg19_1, view_12, permute_9);  arg19_1 = view_12 = permute_9 = None
    view_13: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_3, [8, 3136, 512]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
    mul_9: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
    erf: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_8, add_9);  mul_8 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_3: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_14: "f32[25088, 512]" = torch.ops.aten.view.default(clone_3, [25088, 512]);  clone_3 = None
    permute_10: "f32[512, 64]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_4: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg21_1, view_14, permute_10);  arg21_1 = view_14 = permute_10 = None
    view_15: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_4, [8, 3136, 64]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_4: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_15);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_10: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_6, clone_4);  add_6 = clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_11: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_10, [0, 2, 1]);  add_10 = None
    view_16: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_11, [8, 64, 56, 56]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_2: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_16, arg22_1, arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  arg22_1 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_11: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_2, view_16);  convolution_2 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_18: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_11, [8, 64, 3136]);  add_11 = None
    permute_13: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(permute_13, [2], correction = 0, keepdim = True)
    getitem_19: "f32[8, 3136, 1]" = var_mean_4[0]
    getitem_20: "f32[8, 3136, 1]" = var_mean_4[1];  var_mean_4 = None
    add_12: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-06);  getitem_19 = None
    rsqrt_4: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_4: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(permute_13, getitem_20);  getitem_20 = None
    mul_11: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_12: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_11, arg24_1);  mul_11 = arg24_1 = None
    add_13: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_12, arg25_1);  mul_12 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_19: "f32[25088, 64]" = torch.ops.aten.view.default(add_13, [25088, 64])
    permute_14: "f32[64, 64]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_5: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg27_1, view_19, permute_14);  arg27_1 = view_19 = permute_14 = None
    view_20: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_5, [8, 3136, 64]);  addmm_5 = None
    view_21: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_20, [8, 3136, 1, 64]);  view_20 = None
    permute_15: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_16: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_13, [0, 2, 1]);  add_13 = None
    view_22: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_16, [8, 64, 56, 56]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_3: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_22, arg28_1, arg29_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_22 = arg28_1 = arg29_1 = None
    view_23: "f32[8, 64, 49]" = torch.ops.aten.view.default(convolution_3, [8, 64, 49]);  convolution_3 = None
    permute_17: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_5 = torch.ops.aten.var_mean.correction(permute_17, [2], correction = 0, keepdim = True)
    getitem_21: "f32[8, 49, 1]" = var_mean_5[0]
    getitem_22: "f32[8, 49, 1]" = var_mean_5[1];  var_mean_5 = None
    add_14: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-05);  getitem_21 = None
    rsqrt_5: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_5: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(permute_17, getitem_22);  permute_17 = getitem_22 = None
    mul_13: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_14: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
    add_15: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_14, arg31_1);  mul_14 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_24: "f32[392, 64]" = torch.ops.aten.view.default(add_15, [392, 64]);  add_15 = None
    permute_18: "f32[64, 128]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_6: "f32[392, 128]" = torch.ops.aten.addmm.default(arg33_1, view_24, permute_18);  arg33_1 = view_24 = permute_18 = None
    view_25: "f32[8, 49, 128]" = torch.ops.aten.view.default(addmm_6, [8, 49, 128]);  addmm_6 = None
    view_26: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.view.default(view_25, [8, -1, 2, 1, 64]);  view_25 = None
    permute_19: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_26, [2, 0, 3, 1, 4]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
    getitem_23: "f32[8, 1, 49, 64]" = unbind_1[0]
    getitem_24: "f32[8, 1, 49, 64]" = unbind_1[1];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_15, getitem_23, getitem_24);  permute_15 = getitem_23 = getitem_24 = None
    getitem_25: "f32[8, 1, 3136, 64]" = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_20: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_25, [0, 2, 1, 3]);  getitem_25 = None
    view_27: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_20, [8, 3136, 64]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_28: "f32[25088, 64]" = torch.ops.aten.view.default(view_27, [25088, 64]);  view_27 = None
    permute_21: "f32[64, 64]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    addmm_7: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg35_1, view_28, permute_21);  arg35_1 = view_28 = permute_21 = None
    view_29: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_7, [8, 3136, 64]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_5: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_29);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_16: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_13, clone_5);  permute_13 = clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 3136, 1]" = var_mean_6[0]
    getitem_35: "f32[8, 3136, 1]" = var_mean_6[1];  var_mean_6 = None
    add_17: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_6: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_16, getitem_35);  getitem_35 = None
    mul_15: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_16: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_15, arg36_1);  mul_15 = arg36_1 = None
    add_18: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_16, arg37_1);  mul_16 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[25088, 64]" = torch.ops.aten.view.default(add_18, [25088, 64]);  add_18 = None
    permute_22: "f32[64, 512]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_8: "f32[25088, 512]" = torch.ops.aten.addmm.default(arg39_1, view_30, permute_22);  arg39_1 = view_30 = permute_22 = None
    view_31: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_8, [8, 3136, 512]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_18: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476);  view_31 = None
    erf_1: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_17, add_19);  mul_17 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[25088, 512]" = torch.ops.aten.view.default(clone_6, [25088, 512]);  clone_6 = None
    permute_23: "f32[512, 64]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_9: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg41_1, view_32, permute_23);  arg41_1 = view_32 = permute_23 = None
    view_33: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_9, [8, 3136, 64]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_20: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_16, clone_7);  add_16 = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 3136, 1]" = var_mean_7[0]
    getitem_37: "f32[8, 3136, 1]" = var_mean_7[1];  var_mean_7 = None
    add_21: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_7: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_7: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_20, getitem_37);  getitem_37 = None
    mul_20: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_21: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_20, arg42_1);  mul_20 = arg42_1 = None
    add_22: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_21, arg43_1);  mul_21 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_34: "f32[25088, 64]" = torch.ops.aten.view.default(add_22, [25088, 64])
    permute_24: "f32[64, 64]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    addmm_10: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg45_1, view_34, permute_24);  arg45_1 = view_34 = permute_24 = None
    view_35: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_10, [8, 3136, 64]);  addmm_10 = None
    view_36: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_35, [8, 3136, 1, 64]);  view_35 = None
    permute_25: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_26: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_22, [0, 2, 1]);  add_22 = None
    view_37: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_26, [8, 64, 56, 56]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_4: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_37, arg46_1, arg47_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_37 = arg46_1 = arg47_1 = None
    view_38: "f32[8, 64, 49]" = torch.ops.aten.view.default(convolution_4, [8, 64, 49]);  convolution_4 = None
    permute_27: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_8 = torch.ops.aten.var_mean.correction(permute_27, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 49, 1]" = var_mean_8[0]
    getitem_39: "f32[8, 49, 1]" = var_mean_8[1];  var_mean_8 = None
    add_23: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_8: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_8: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(permute_27, getitem_39);  permute_27 = getitem_39 = None
    mul_22: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_23: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_22, arg48_1);  mul_22 = arg48_1 = None
    add_24: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_23, arg49_1);  mul_23 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_39: "f32[392, 64]" = torch.ops.aten.view.default(add_24, [392, 64]);  add_24 = None
    permute_28: "f32[64, 128]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    addmm_11: "f32[392, 128]" = torch.ops.aten.addmm.default(arg51_1, view_39, permute_28);  arg51_1 = view_39 = permute_28 = None
    view_40: "f32[8, 49, 128]" = torch.ops.aten.view.default(addmm_11, [8, 49, 128]);  addmm_11 = None
    view_41: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.view.default(view_40, [8, -1, 2, 1, 64]);  view_40 = None
    permute_29: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_29);  permute_29 = None
    getitem_40: "f32[8, 1, 49, 64]" = unbind_2[0]
    getitem_41: "f32[8, 1, 49, 64]" = unbind_2[1];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_25, getitem_40, getitem_41);  permute_25 = getitem_40 = getitem_41 = None
    getitem_42: "f32[8, 1, 3136, 64]" = _scaled_dot_product_flash_attention_2[0];  _scaled_dot_product_flash_attention_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_30: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
    view_42: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_30, [8, 3136, 64]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_43: "f32[25088, 64]" = torch.ops.aten.view.default(view_42, [25088, 64]);  view_42 = None
    permute_31: "f32[64, 64]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_12: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg53_1, view_43, permute_31);  arg53_1 = view_43 = permute_31 = None
    view_44: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_12, [8, 3136, 64]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_8: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_25: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_20, clone_8);  add_20 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_51: "f32[8, 3136, 1]" = var_mean_9[0]
    getitem_52: "f32[8, 3136, 1]" = var_mean_9[1];  var_mean_9 = None
    add_26: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_51, 1e-06);  getitem_51 = None
    rsqrt_9: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_9: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_25, getitem_52);  getitem_52 = None
    mul_24: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_25: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_24, arg54_1);  mul_24 = arg54_1 = None
    add_27: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_25, arg55_1);  mul_25 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[25088, 64]" = torch.ops.aten.view.default(add_27, [25088, 64]);  add_27 = None
    permute_32: "f32[64, 512]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_13: "f32[25088, 512]" = torch.ops.aten.addmm.default(arg57_1, view_45, permute_32);  arg57_1 = view_45 = permute_32 = None
    view_46: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_13, [8, 3136, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_27: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476);  view_46 = None
    erf_2: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_28: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_28: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_26, add_28);  mul_26 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[25088, 512]" = torch.ops.aten.view.default(clone_9, [25088, 512]);  clone_9 = None
    permute_33: "f32[512, 64]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_14: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg59_1, view_47, permute_33);  arg59_1 = view_47 = permute_33 = None
    view_48: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_14, [8, 3136, 64]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_29: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_25, clone_10);  add_25 = clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    view_49: "f32[8, 56, 56, 64]" = torch.ops.aten.view.default(add_29, [8, 56, 56, -1]);  add_29 = None
    permute_34: "f32[8, 64, 56, 56]" = torch.ops.aten.permute.default(view_49, [0, 3, 1, 2]);  view_49 = None
    clone_11: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution_5: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(clone_11, arg60_1, arg61_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_11 = arg60_1 = arg61_1 = None
    view_50: "f32[8, 128, 784]" = torch.ops.aten.view.default(convolution_5, [8, 128, 784]);  convolution_5 = None
    permute_35: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_12: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_53: "f32[8, 784, 1]" = var_mean_10[0]
    getitem_54: "f32[8, 784, 1]" = var_mean_10[1];  var_mean_10 = None
    add_30: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
    rsqrt_10: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_10: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_12, getitem_54);  clone_12 = getitem_54 = None
    mul_29: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_30: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
    add_31: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_30, arg63_1);  mul_30 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_13: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_13, [2], correction = 0, keepdim = True)
    getitem_55: "f32[8, 784, 1]" = var_mean_11[0]
    getitem_56: "f32[8, 784, 1]" = var_mean_11[1];  var_mean_11 = None
    add_32: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_11: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_11: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_13, getitem_56);  getitem_56 = None
    mul_31: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_32: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_31, arg64_1);  mul_31 = arg64_1 = None
    add_33: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_32, arg65_1);  mul_32 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_51: "f32[6272, 128]" = torch.ops.aten.view.default(add_33, [6272, 128])
    permute_36: "f32[128, 128]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_15: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg67_1, view_51, permute_36);  arg67_1 = view_51 = permute_36 = None
    view_52: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_15, [8, 784, 128]);  addmm_15 = None
    view_53: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_52, [8, 784, 2, 64]);  view_52 = None
    permute_37: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_38: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_33, [0, 2, 1]);  add_33 = None
    view_54: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_38, [8, 128, 28, 28]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_6: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_54, arg68_1, arg69_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_54 = arg68_1 = arg69_1 = None
    view_55: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_6, [8, 128, 49]);  convolution_6 = None
    permute_39: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_12 = torch.ops.aten.var_mean.correction(permute_39, [2], correction = 0, keepdim = True)
    getitem_57: "f32[8, 49, 1]" = var_mean_12[0]
    getitem_58: "f32[8, 49, 1]" = var_mean_12[1];  var_mean_12 = None
    add_34: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-05);  getitem_57 = None
    rsqrt_12: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_12: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_39, getitem_58);  permute_39 = getitem_58 = None
    mul_33: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_34: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_33, arg70_1);  mul_33 = arg70_1 = None
    add_35: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_34, arg71_1);  mul_34 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_56: "f32[392, 128]" = torch.ops.aten.view.default(add_35, [392, 128]);  add_35 = None
    permute_40: "f32[128, 256]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_16: "f32[392, 256]" = torch.ops.aten.addmm.default(arg73_1, view_56, permute_40);  arg73_1 = view_56 = permute_40 = None
    view_57: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_16, [8, 49, 256]);  addmm_16 = None
    view_58: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_57, [8, -1, 2, 2, 64]);  view_57 = None
    permute_41: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_58, [2, 0, 3, 1, 4]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_41);  permute_41 = None
    getitem_59: "f32[8, 2, 49, 64]" = unbind_3[0]
    getitem_60: "f32[8, 2, 49, 64]" = unbind_3[1];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_37, getitem_59, getitem_60);  permute_37 = getitem_59 = getitem_60 = None
    getitem_61: "f32[8, 2, 784, 64]" = _scaled_dot_product_flash_attention_3[0];  _scaled_dot_product_flash_attention_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_42: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
    view_59: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_42, [8, 784, 128]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_60: "f32[6272, 128]" = torch.ops.aten.view.default(view_59, [6272, 128]);  view_59 = None
    permute_43: "f32[128, 128]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_17: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg75_1, view_60, permute_43);  arg75_1 = view_60 = permute_43 = None
    view_61: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_17, [8, 784, 128]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_14: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_61);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_36: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(clone_13, clone_14);  clone_13 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 784, 1]" = var_mean_13[0]
    getitem_71: "f32[8, 784, 1]" = var_mean_13[1];  var_mean_13 = None
    add_37: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_13: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_13: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_36, getitem_71);  getitem_71 = None
    mul_35: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_36: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_35, arg76_1);  mul_35 = arg76_1 = None
    add_38: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_62: "f32[6272, 128]" = torch.ops.aten.view.default(add_38, [6272, 128]);  add_38 = None
    permute_44: "f32[128, 1024]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_18: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg79_1, view_62, permute_44);  arg79_1 = view_62 = permute_44 = None
    view_63: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_18, [8, 784, 1024]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_38: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_3: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_37, add_39);  mul_37 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_64: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_15, [6272, 1024]);  clone_15 = None
    permute_45: "f32[1024, 128]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_19: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg81_1, view_64, permute_45);  arg81_1 = view_64 = permute_45 = None
    view_65: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_19, [8, 784, 128]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_40: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_36, clone_16);  add_36 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_46: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    view_66: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_46, [8, 128, 28, 28]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_7: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_66, arg82_1, arg83_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg82_1 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_41: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_7, view_66);  convolution_7 = view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_68: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_41, [8, 128, 784]);  add_41 = None
    permute_48: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(permute_48, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 784, 1]" = var_mean_14[0]
    getitem_73: "f32[8, 784, 1]" = var_mean_14[1];  var_mean_14 = None
    add_42: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_14: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_14: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(permute_48, getitem_73);  getitem_73 = None
    mul_40: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_41: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_40, arg84_1);  mul_40 = arg84_1 = None
    add_43: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_41, arg85_1);  mul_41 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_69: "f32[6272, 128]" = torch.ops.aten.view.default(add_43, [6272, 128])
    permute_49: "f32[128, 128]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_20: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg87_1, view_69, permute_49);  arg87_1 = view_69 = permute_49 = None
    view_70: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_20, [8, 784, 128]);  addmm_20 = None
    view_71: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_70, [8, 784, 2, 64]);  view_70 = None
    permute_50: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_51: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_43, [0, 2, 1]);  add_43 = None
    view_72: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_51, [8, 128, 28, 28]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_8: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_72, arg88_1, arg89_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_72 = arg88_1 = arg89_1 = None
    view_73: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_8, [8, 128, 49]);  convolution_8 = None
    permute_52: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_15 = torch.ops.aten.var_mean.correction(permute_52, [2], correction = 0, keepdim = True)
    getitem_74: "f32[8, 49, 1]" = var_mean_15[0]
    getitem_75: "f32[8, 49, 1]" = var_mean_15[1];  var_mean_15 = None
    add_44: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_15: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_15: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_52, getitem_75);  permute_52 = getitem_75 = None
    mul_42: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_43: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_42, arg90_1);  mul_42 = arg90_1 = None
    add_45: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_43, arg91_1);  mul_43 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_74: "f32[392, 128]" = torch.ops.aten.view.default(add_45, [392, 128]);  add_45 = None
    permute_53: "f32[128, 256]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_21: "f32[392, 256]" = torch.ops.aten.addmm.default(arg93_1, view_74, permute_53);  arg93_1 = view_74 = permute_53 = None
    view_75: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_21, [8, 49, 256]);  addmm_21 = None
    view_76: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_75, [8, -1, 2, 2, 64]);  view_75 = None
    permute_54: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_76, [2, 0, 3, 1, 4]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_54);  permute_54 = None
    getitem_76: "f32[8, 2, 49, 64]" = unbind_4[0]
    getitem_77: "f32[8, 2, 49, 64]" = unbind_4[1];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_50, getitem_76, getitem_77);  permute_50 = getitem_76 = getitem_77 = None
    getitem_78: "f32[8, 2, 784, 64]" = _scaled_dot_product_flash_attention_4[0];  _scaled_dot_product_flash_attention_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_55: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_78, [0, 2, 1, 3]);  getitem_78 = None
    view_77: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_55, [8, 784, 128]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_78: "f32[6272, 128]" = torch.ops.aten.view.default(view_77, [6272, 128]);  view_77 = None
    permute_56: "f32[128, 128]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_22: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg95_1, view_78, permute_56);  arg95_1 = view_78 = permute_56 = None
    view_79: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_22, [8, 784, 128]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_17: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_79);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_46: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_48, clone_17);  permute_48 = clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_87: "f32[8, 784, 1]" = var_mean_16[0]
    getitem_88: "f32[8, 784, 1]" = var_mean_16[1];  var_mean_16 = None
    add_47: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_87, 1e-06);  getitem_87 = None
    rsqrt_16: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_16: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_46, getitem_88);  getitem_88 = None
    mul_44: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_45: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_44, arg96_1);  mul_44 = arg96_1 = None
    add_48: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_45, arg97_1);  mul_45 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_80: "f32[6272, 128]" = torch.ops.aten.view.default(add_48, [6272, 128]);  add_48 = None
    permute_57: "f32[128, 1024]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_23: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg99_1, view_80, permute_57);  arg99_1 = view_80 = permute_57 = None
    view_81: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_23, [8, 784, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
    mul_47: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476);  view_81 = None
    erf_4: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_48: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_18: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_82: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_18, [6272, 1024]);  clone_18 = None
    permute_58: "f32[1024, 128]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_24: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg101_1, view_82, permute_58);  arg101_1 = view_82 = permute_58 = None
    view_83: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_24, [8, 784, 128]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_19: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_50: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_46, clone_19);  add_46 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_89: "f32[8, 784, 1]" = var_mean_17[0]
    getitem_90: "f32[8, 784, 1]" = var_mean_17[1];  var_mean_17 = None
    add_51: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-06);  getitem_89 = None
    rsqrt_17: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_17: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_50, getitem_90);  getitem_90 = None
    mul_49: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_50: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_49, arg102_1);  mul_49 = arg102_1 = None
    add_52: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_50, arg103_1);  mul_50 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_84: "f32[6272, 128]" = torch.ops.aten.view.default(add_52, [6272, 128])
    permute_59: "f32[128, 128]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_25: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg105_1, view_84, permute_59);  arg105_1 = view_84 = permute_59 = None
    view_85: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_25, [8, 784, 128]);  addmm_25 = None
    view_86: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_85, [8, 784, 2, 64]);  view_85 = None
    permute_60: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_61: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None
    view_87: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_61, [8, 128, 28, 28]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_9: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_87, arg106_1, arg107_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_87 = arg106_1 = arg107_1 = None
    view_88: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_9, [8, 128, 49]);  convolution_9 = None
    permute_62: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_18 = torch.ops.aten.var_mean.correction(permute_62, [2], correction = 0, keepdim = True)
    getitem_91: "f32[8, 49, 1]" = var_mean_18[0]
    getitem_92: "f32[8, 49, 1]" = var_mean_18[1];  var_mean_18 = None
    add_53: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
    rsqrt_18: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_18: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_62, getitem_92);  permute_62 = getitem_92 = None
    mul_51: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_52: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_51, arg108_1);  mul_51 = arg108_1 = None
    add_54: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_89: "f32[392, 128]" = torch.ops.aten.view.default(add_54, [392, 128]);  add_54 = None
    permute_63: "f32[128, 256]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_26: "f32[392, 256]" = torch.ops.aten.addmm.default(arg111_1, view_89, permute_63);  arg111_1 = view_89 = permute_63 = None
    view_90: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_26, [8, 49, 256]);  addmm_26 = None
    view_91: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_90, [8, -1, 2, 2, 64]);  view_90 = None
    permute_64: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_91, [2, 0, 3, 1, 4]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_64);  permute_64 = None
    getitem_93: "f32[8, 2, 49, 64]" = unbind_5[0]
    getitem_94: "f32[8, 2, 49, 64]" = unbind_5[1];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_60, getitem_93, getitem_94);  permute_60 = getitem_93 = getitem_94 = None
    getitem_95: "f32[8, 2, 784, 64]" = _scaled_dot_product_flash_attention_5[0];  _scaled_dot_product_flash_attention_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_65: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_95, [0, 2, 1, 3]);  getitem_95 = None
    view_92: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_65, [8, 784, 128]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_93: "f32[6272, 128]" = torch.ops.aten.view.default(view_92, [6272, 128]);  view_92 = None
    permute_66: "f32[128, 128]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_27: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg113_1, view_93, permute_66);  arg113_1 = view_93 = permute_66 = None
    view_94: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_27, [8, 784, 128]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_20: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_94);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_55: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_50, clone_20);  add_50 = clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 784, 1]" = var_mean_19[0]
    getitem_105: "f32[8, 784, 1]" = var_mean_19[1];  var_mean_19 = None
    add_56: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_19: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_19: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_55, getitem_105);  getitem_105 = None
    mul_53: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_54: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_53, arg114_1);  mul_53 = arg114_1 = None
    add_57: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_54, arg115_1);  mul_54 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_95: "f32[6272, 128]" = torch.ops.aten.view.default(add_57, [6272, 128]);  add_57 = None
    permute_67: "f32[128, 1024]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_28: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg117_1, view_95, permute_67);  arg117_1 = view_95 = permute_67 = None
    view_96: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_28, [8, 784, 1024]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_55: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, 0.5)
    mul_56: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476);  view_96 = None
    erf_5: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_58: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_57: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_55, add_58);  mul_55 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_21: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_97: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_21, [6272, 1024]);  clone_21 = None
    permute_68: "f32[1024, 128]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_29: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg119_1, view_97, permute_68);  arg119_1 = view_97 = permute_68 = None
    view_98: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_29, [8, 784, 128]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_22: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_98);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_59: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_55, clone_22);  add_55 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 784, 1]" = var_mean_20[0]
    getitem_107: "f32[8, 784, 1]" = var_mean_20[1];  var_mean_20 = None
    add_60: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_20: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_20: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_59, getitem_107);  getitem_107 = None
    mul_58: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_59: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_58, arg120_1);  mul_58 = arg120_1 = None
    add_61: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_59, arg121_1);  mul_59 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_99: "f32[6272, 128]" = torch.ops.aten.view.default(add_61, [6272, 128])
    permute_69: "f32[128, 128]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_30: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg123_1, view_99, permute_69);  arg123_1 = view_99 = permute_69 = None
    view_100: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_30, [8, 784, 128]);  addmm_30 = None
    view_101: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_100, [8, 784, 2, 64]);  view_100 = None
    permute_70: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_71: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_61, [0, 2, 1]);  add_61 = None
    view_102: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_71, [8, 128, 28, 28]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_10: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_102, arg124_1, arg125_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_102 = arg124_1 = arg125_1 = None
    view_103: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_10, [8, 128, 49]);  convolution_10 = None
    permute_72: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_21 = torch.ops.aten.var_mean.correction(permute_72, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 49, 1]" = var_mean_21[0]
    getitem_109: "f32[8, 49, 1]" = var_mean_21[1];  var_mean_21 = None
    add_62: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_21: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_21: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_72, getitem_109);  permute_72 = getitem_109 = None
    mul_60: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_61: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_60, arg126_1);  mul_60 = arg126_1 = None
    add_63: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_61, arg127_1);  mul_61 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_104: "f32[392, 128]" = torch.ops.aten.view.default(add_63, [392, 128]);  add_63 = None
    permute_73: "f32[128, 256]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_31: "f32[392, 256]" = torch.ops.aten.addmm.default(arg129_1, view_104, permute_73);  arg129_1 = view_104 = permute_73 = None
    view_105: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_31, [8, 49, 256]);  addmm_31 = None
    view_106: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_105, [8, -1, 2, 2, 64]);  view_105 = None
    permute_74: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_106, [2, 0, 3, 1, 4]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_110: "f32[8, 2, 49, 64]" = unbind_6[0]
    getitem_111: "f32[8, 2, 49, 64]" = unbind_6[1];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_70, getitem_110, getitem_111);  permute_70 = getitem_110 = getitem_111 = None
    getitem_112: "f32[8, 2, 784, 64]" = _scaled_dot_product_flash_attention_6[0];  _scaled_dot_product_flash_attention_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_75: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None
    view_107: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_75, [8, 784, 128]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_108: "f32[6272, 128]" = torch.ops.aten.view.default(view_107, [6272, 128]);  view_107 = None
    permute_76: "f32[128, 128]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_32: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg131_1, view_108, permute_76);  arg131_1 = view_108 = permute_76 = None
    view_109: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_32, [8, 784, 128]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_23: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_64: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_59, clone_23);  add_59 = clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_22 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 784, 1]" = var_mean_22[0]
    getitem_122: "f32[8, 784, 1]" = var_mean_22[1];  var_mean_22 = None
    add_65: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_22: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_22: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_64, getitem_122);  getitem_122 = None
    mul_62: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_63: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_62, arg132_1);  mul_62 = arg132_1 = None
    add_66: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_63, arg133_1);  mul_63 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_110: "f32[6272, 128]" = torch.ops.aten.view.default(add_66, [6272, 128]);  add_66 = None
    permute_77: "f32[128, 1024]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_33: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg135_1, view_110, permute_77);  arg135_1 = view_110 = permute_77 = None
    view_111: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_33, [8, 784, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_64: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_65: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
    erf_6: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_67: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_66: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_64, add_67);  mul_64 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_24: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_112: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_24, [6272, 1024]);  clone_24 = None
    permute_78: "f32[1024, 128]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_34: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg137_1, view_112, permute_78);  arg137_1 = view_112 = permute_78 = None
    view_113: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_34, [8, 784, 128]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_25: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_68: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_64, clone_25);  add_64 = clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    view_114: "f32[8, 28, 28, 128]" = torch.ops.aten.view.default(add_68, [8, 28, 28, -1]);  add_68 = None
    permute_79: "f32[8, 128, 28, 28]" = torch.ops.aten.permute.default(view_114, [0, 3, 1, 2]);  view_114 = None
    clone_26: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution_11: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(clone_26, arg138_1, arg139_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_26 = arg138_1 = arg139_1 = None
    view_115: "f32[8, 320, 196]" = torch.ops.aten.view.default(convolution_11, [8, 320, 196]);  convolution_11 = None
    permute_80: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_27: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_27, [2], correction = 0, keepdim = True)
    getitem_123: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_124: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_69: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_123, 1e-05);  getitem_123 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_23: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_27, getitem_124);  clone_27 = getitem_124 = None
    mul_67: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_68: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_67, arg140_1);  mul_67 = arg140_1 = None
    add_70: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_68, arg141_1);  mul_68 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_28: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_70);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_125: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_126: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_71: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-06);  getitem_125 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_24: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_28, getitem_126);  getitem_126 = None
    mul_69: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_70: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
    add_72: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_70, arg143_1);  mul_70 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_116: "f32[1568, 320]" = torch.ops.aten.view.default(add_72, [1568, 320])
    permute_81: "f32[320, 320]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_35: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg145_1, view_116, permute_81);  arg145_1 = view_116 = permute_81 = None
    view_117: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_35, [8, 196, 320]);  addmm_35 = None
    view_118: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_117, [8, 196, 5, 64]);  view_117 = None
    permute_82: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_83: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_72, [0, 2, 1]);  add_72 = None
    view_119: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_83, [8, 320, 14, 14]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_12: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_119, arg146_1, arg147_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_119 = arg146_1 = arg147_1 = None
    view_120: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_12, [8, 320, 49]);  convolution_12 = None
    permute_84: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_25 = torch.ops.aten.var_mean.correction(permute_84, [2], correction = 0, keepdim = True)
    getitem_127: "f32[8, 49, 1]" = var_mean_25[0]
    getitem_128: "f32[8, 49, 1]" = var_mean_25[1];  var_mean_25 = None
    add_73: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_127, 1e-05);  getitem_127 = None
    rsqrt_25: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_25: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_84, getitem_128);  permute_84 = getitem_128 = None
    mul_71: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
    mul_72: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_71, arg148_1);  mul_71 = arg148_1 = None
    add_74: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_72, arg149_1);  mul_72 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_121: "f32[392, 320]" = torch.ops.aten.view.default(add_74, [392, 320]);  add_74 = None
    permute_85: "f32[320, 640]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_36: "f32[392, 640]" = torch.ops.aten.addmm.default(arg151_1, view_121, permute_85);  arg151_1 = view_121 = permute_85 = None
    view_122: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_36, [8, 49, 640]);  addmm_36 = None
    view_123: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_122, [8, -1, 2, 5, 64]);  view_122 = None
    permute_86: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_86);  permute_86 = None
    getitem_129: "f32[8, 5, 49, 64]" = unbind_7[0]
    getitem_130: "f32[8, 5, 49, 64]" = unbind_7[1];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_82, getitem_129, getitem_130);  permute_82 = getitem_129 = getitem_130 = None
    getitem_131: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_7[0];  _scaled_dot_product_flash_attention_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_87: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3]);  getitem_131 = None
    view_124: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_87, [8, 196, 320]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_125: "f32[1568, 320]" = torch.ops.aten.view.default(view_124, [1568, 320]);  view_124 = None
    permute_88: "f32[320, 320]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_37: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg153_1, view_125, permute_88);  arg153_1 = view_125 = permute_88 = None
    view_126: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_37, [8, 196, 320]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_29: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_75: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(clone_28, clone_29);  clone_28 = clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_26 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_141: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_76: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_26: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_75, getitem_141);  getitem_141 = None
    mul_73: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
    mul_74: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_73, arg154_1);  mul_73 = arg154_1 = None
    add_77: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_74, arg155_1);  mul_74 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[1568, 320]" = torch.ops.aten.view.default(add_77, [1568, 320]);  add_77 = None
    permute_89: "f32[320, 1280]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_38: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg157_1, view_127, permute_89);  arg157_1 = view_127 = permute_89 = None
    view_128: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_38, [8, 196, 1280]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_76: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476);  view_128 = None
    erf_7: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_78: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_77: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_75, add_78);  mul_75 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_30: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_77);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_30, [1568, 1280]);  clone_30 = None
    permute_90: "f32[1280, 320]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_39: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg159_1, view_129, permute_90);  arg159_1 = view_129 = permute_90 = None
    view_130: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_39, [8, 196, 320]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_31: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_130);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_79: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_75, clone_31);  add_75 = clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_91: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_79, [0, 2, 1]);  add_79 = None
    view_131: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_91, [8, 320, 14, 14]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_13: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_131, arg160_1, arg161_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  arg160_1 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_80: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_13, view_131);  convolution_13 = view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_133: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_80, [8, 320, 196]);  add_80 = None
    permute_93: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(permute_93, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_143: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_27: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(permute_93, getitem_143);  getitem_143 = None
    mul_78: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
    mul_79: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_78, arg162_1);  mul_78 = arg162_1 = None
    add_82: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_79, arg163_1);  mul_79 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_134: "f32[1568, 320]" = torch.ops.aten.view.default(add_82, [1568, 320])
    permute_94: "f32[320, 320]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_40: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg165_1, view_134, permute_94);  arg165_1 = view_134 = permute_94 = None
    view_135: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_40, [8, 196, 320]);  addmm_40 = None
    view_136: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_135, [8, 196, 5, 64]);  view_135 = None
    permute_95: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_96: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    view_137: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_96, [8, 320, 14, 14]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_14: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_137, arg166_1, arg167_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_137 = arg166_1 = arg167_1 = None
    view_138: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_14, [8, 320, 49]);  convolution_14 = None
    permute_97: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_28 = torch.ops.aten.var_mean.correction(permute_97, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 49, 1]" = var_mean_28[0]
    getitem_145: "f32[8, 49, 1]" = var_mean_28[1];  var_mean_28 = None
    add_83: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
    rsqrt_28: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_28: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_97, getitem_145);  permute_97 = getitem_145 = None
    mul_80: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
    mul_81: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_80, arg168_1);  mul_80 = arg168_1 = None
    add_84: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_81, arg169_1);  mul_81 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_139: "f32[392, 320]" = torch.ops.aten.view.default(add_84, [392, 320]);  add_84 = None
    permute_98: "f32[320, 640]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_41: "f32[392, 640]" = torch.ops.aten.addmm.default(arg171_1, view_139, permute_98);  arg171_1 = view_139 = permute_98 = None
    view_140: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_41, [8, 49, 640]);  addmm_41 = None
    view_141: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_140, [8, -1, 2, 5, 64]);  view_140 = None
    permute_99: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_141, [2, 0, 3, 1, 4]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_99);  permute_99 = None
    getitem_146: "f32[8, 5, 49, 64]" = unbind_8[0]
    getitem_147: "f32[8, 5, 49, 64]" = unbind_8[1];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_95, getitem_146, getitem_147);  permute_95 = getitem_146 = getitem_147 = None
    getitem_148: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_8[0];  _scaled_dot_product_flash_attention_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_100: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
    view_142: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_100, [8, 196, 320]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_143: "f32[1568, 320]" = torch.ops.aten.view.default(view_142, [1568, 320]);  view_142 = None
    permute_101: "f32[320, 320]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_42: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg173_1, view_143, permute_101);  arg173_1 = view_143 = permute_101 = None
    view_144: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_42, [8, 196, 320]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_32: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_85: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_93, clone_32);  permute_93 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_29 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_157: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_158: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_86: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_157, 1e-06);  getitem_157 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_29: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_85, getitem_158);  getitem_158 = None
    mul_82: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
    mul_83: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_82, arg174_1);  mul_82 = arg174_1 = None
    add_87: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_83, arg175_1);  mul_83 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[1568, 320]" = torch.ops.aten.view.default(add_87, [1568, 320]);  add_87 = None
    permute_102: "f32[320, 1280]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_43: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg177_1, view_145, permute_102);  arg177_1 = view_145 = permute_102 = None
    view_146: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_43, [8, 196, 1280]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_84: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_85: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476);  view_146 = None
    erf_8: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_88: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_86: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_33: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_86);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_33, [1568, 1280]);  clone_33 = None
    permute_103: "f32[1280, 320]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_44: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg179_1, view_147, permute_103);  arg179_1 = view_147 = permute_103 = None
    view_148: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_44, [8, 196, 320]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_34: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_148);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_89: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_85, clone_34);  add_85 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_30 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_159: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_160: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_90: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_159, 1e-06);  getitem_159 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_30: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_89, getitem_160);  getitem_160 = None
    mul_87: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
    mul_88: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_87, arg180_1);  mul_87 = arg180_1 = None
    add_91: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_88, arg181_1);  mul_88 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_149: "f32[1568, 320]" = torch.ops.aten.view.default(add_91, [1568, 320])
    permute_104: "f32[320, 320]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_45: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg183_1, view_149, permute_104);  arg183_1 = view_149 = permute_104 = None
    view_150: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_45, [8, 196, 320]);  addmm_45 = None
    view_151: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_150, [8, 196, 5, 64]);  view_150 = None
    permute_105: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_106: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_91, [0, 2, 1]);  add_91 = None
    view_152: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_106, [8, 320, 14, 14]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_15: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_152, arg184_1, arg185_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_152 = arg184_1 = arg185_1 = None
    view_153: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_15, [8, 320, 49]);  convolution_15 = None
    permute_107: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_31 = torch.ops.aten.var_mean.correction(permute_107, [2], correction = 0, keepdim = True)
    getitem_161: "f32[8, 49, 1]" = var_mean_31[0]
    getitem_162: "f32[8, 49, 1]" = var_mean_31[1];  var_mean_31 = None
    add_92: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_161, 1e-05);  getitem_161 = None
    rsqrt_31: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_31: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_107, getitem_162);  permute_107 = getitem_162 = None
    mul_89: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
    mul_90: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_89, arg186_1);  mul_89 = arg186_1 = None
    add_93: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_90, arg187_1);  mul_90 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_154: "f32[392, 320]" = torch.ops.aten.view.default(add_93, [392, 320]);  add_93 = None
    permute_108: "f32[320, 640]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_46: "f32[392, 640]" = torch.ops.aten.addmm.default(arg189_1, view_154, permute_108);  arg189_1 = view_154 = permute_108 = None
    view_155: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_46, [8, 49, 640]);  addmm_46 = None
    view_156: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_155, [8, -1, 2, 5, 64]);  view_155 = None
    permute_109: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_156, [2, 0, 3, 1, 4]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_109);  permute_109 = None
    getitem_163: "f32[8, 5, 49, 64]" = unbind_9[0]
    getitem_164: "f32[8, 5, 49, 64]" = unbind_9[1];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_105, getitem_163, getitem_164);  permute_105 = getitem_163 = getitem_164 = None
    getitem_165: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_9[0];  _scaled_dot_product_flash_attention_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_110: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
    view_157: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_110, [8, 196, 320]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_158: "f32[1568, 320]" = torch.ops.aten.view.default(view_157, [1568, 320]);  view_157 = None
    permute_111: "f32[320, 320]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_47: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg191_1, view_158, permute_111);  arg191_1 = view_158 = permute_111 = None
    view_159: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_47, [8, 196, 320]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_35: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_159);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_94: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_89, clone_35);  add_89 = clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_32 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_174: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_175: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_95: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_32: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_94, getitem_175);  getitem_175 = None
    mul_91: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
    mul_92: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_91, arg192_1);  mul_91 = arg192_1 = None
    add_96: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_92, arg193_1);  mul_92 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_160: "f32[1568, 320]" = torch.ops.aten.view.default(add_96, [1568, 320]);  add_96 = None
    permute_112: "f32[320, 1280]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_48: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg195_1, view_160, permute_112);  arg195_1 = view_160 = permute_112 = None
    view_161: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_48, [8, 196, 1280]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    mul_94: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476);  view_161 = None
    erf_9: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_97: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_95: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_93, add_97);  mul_93 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_36: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_162: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_36, [1568, 1280]);  clone_36 = None
    permute_113: "f32[1280, 320]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    addmm_49: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg197_1, view_162, permute_113);  arg197_1 = view_162 = permute_113 = None
    view_163: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_49, [8, 196, 320]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_37: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_98: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_94, clone_37);  add_94 = clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_33 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_177: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_99: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_33: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_98, getitem_177);  getitem_177 = None
    mul_96: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
    mul_97: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_96, arg198_1);  mul_96 = arg198_1 = None
    add_100: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_97, arg199_1);  mul_97 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_164: "f32[1568, 320]" = torch.ops.aten.view.default(add_100, [1568, 320])
    permute_114: "f32[320, 320]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    addmm_50: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg201_1, view_164, permute_114);  arg201_1 = view_164 = permute_114 = None
    view_165: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_50, [8, 196, 320]);  addmm_50 = None
    view_166: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_165, [8, 196, 5, 64]);  view_165 = None
    permute_115: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_116: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    view_167: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_116, [8, 320, 14, 14]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_16: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_167, arg202_1, arg203_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_167 = arg202_1 = arg203_1 = None
    view_168: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_16, [8, 320, 49]);  convolution_16 = None
    permute_117: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_34 = torch.ops.aten.var_mean.correction(permute_117, [2], correction = 0, keepdim = True)
    getitem_178: "f32[8, 49, 1]" = var_mean_34[0]
    getitem_179: "f32[8, 49, 1]" = var_mean_34[1];  var_mean_34 = None
    add_101: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
    rsqrt_34: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_34: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_117, getitem_179);  permute_117 = getitem_179 = None
    mul_98: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
    mul_99: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_98, arg204_1);  mul_98 = arg204_1 = None
    add_102: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_99, arg205_1);  mul_99 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_169: "f32[392, 320]" = torch.ops.aten.view.default(add_102, [392, 320]);  add_102 = None
    permute_118: "f32[320, 640]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_51: "f32[392, 640]" = torch.ops.aten.addmm.default(arg207_1, view_169, permute_118);  arg207_1 = view_169 = permute_118 = None
    view_170: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_51, [8, 49, 640]);  addmm_51 = None
    view_171: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_170, [8, -1, 2, 5, 64]);  view_170 = None
    permute_119: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_171, [2, 0, 3, 1, 4]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_180: "f32[8, 5, 49, 64]" = unbind_10[0]
    getitem_181: "f32[8, 5, 49, 64]" = unbind_10[1];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_115, getitem_180, getitem_181);  permute_115 = getitem_180 = getitem_181 = None
    getitem_182: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_10[0];  _scaled_dot_product_flash_attention_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_120: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_182, [0, 2, 1, 3]);  getitem_182 = None
    view_172: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_120, [8, 196, 320]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_173: "f32[1568, 320]" = torch.ops.aten.view.default(view_172, [1568, 320]);  view_172 = None
    permute_121: "f32[320, 320]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_52: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg209_1, view_173, permute_121);  arg209_1 = view_173 = permute_121 = None
    view_174: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_52, [8, 196, 320]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_38: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_174);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_103: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_98, clone_38);  add_98 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_35 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_191: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_192: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_104: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_191, 1e-06);  getitem_191 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_35: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_103, getitem_192);  getitem_192 = None
    mul_100: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
    mul_101: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_100, arg210_1);  mul_100 = arg210_1 = None
    add_105: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_101, arg211_1);  mul_101 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_175: "f32[1568, 320]" = torch.ops.aten.view.default(add_105, [1568, 320]);  add_105 = None
    permute_122: "f32[320, 1280]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    addmm_53: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg213_1, view_175, permute_122);  arg213_1 = view_175 = permute_122 = None
    view_176: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_53, [8, 196, 1280]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_102: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_103: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476);  view_176 = None
    erf_10: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_106: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_104: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_102, add_106);  mul_102 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_104);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_177: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_39, [1568, 1280]);  clone_39 = None
    permute_123: "f32[1280, 320]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_54: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg215_1, view_177, permute_123);  arg215_1 = view_177 = permute_123 = None
    view_178: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_54, [8, 196, 320]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_107: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_103, clone_40);  add_103 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_36 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_193: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_194: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_108: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_193, 1e-06);  getitem_193 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_36: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_107, getitem_194);  getitem_194 = None
    mul_105: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
    mul_106: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_105, arg216_1);  mul_105 = arg216_1 = None
    add_109: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_106, arg217_1);  mul_106 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_179: "f32[1568, 320]" = torch.ops.aten.view.default(add_109, [1568, 320])
    permute_124: "f32[320, 320]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    addmm_55: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg219_1, view_179, permute_124);  arg219_1 = view_179 = permute_124 = None
    view_180: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_55, [8, 196, 320]);  addmm_55 = None
    view_181: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_180, [8, 196, 5, 64]);  view_180 = None
    permute_125: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_126: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_109, [0, 2, 1]);  add_109 = None
    view_182: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_126, [8, 320, 14, 14]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_17: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_182, arg220_1, arg221_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_182 = arg220_1 = arg221_1 = None
    view_183: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_17, [8, 320, 49]);  convolution_17 = None
    permute_127: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_37 = torch.ops.aten.var_mean.correction(permute_127, [2], correction = 0, keepdim = True)
    getitem_195: "f32[8, 49, 1]" = var_mean_37[0]
    getitem_196: "f32[8, 49, 1]" = var_mean_37[1];  var_mean_37 = None
    add_110: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_195, 1e-05);  getitem_195 = None
    rsqrt_37: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_127, getitem_196);  permute_127 = getitem_196 = None
    mul_107: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
    mul_108: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_107, arg222_1);  mul_107 = arg222_1 = None
    add_111: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_108, arg223_1);  mul_108 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_184: "f32[392, 320]" = torch.ops.aten.view.default(add_111, [392, 320]);  add_111 = None
    permute_128: "f32[320, 640]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    addmm_56: "f32[392, 640]" = torch.ops.aten.addmm.default(arg225_1, view_184, permute_128);  arg225_1 = view_184 = permute_128 = None
    view_185: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_56, [8, 49, 640]);  addmm_56 = None
    view_186: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_185, [8, -1, 2, 5, 64]);  view_185 = None
    permute_129: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_129);  permute_129 = None
    getitem_197: "f32[8, 5, 49, 64]" = unbind_11[0]
    getitem_198: "f32[8, 5, 49, 64]" = unbind_11[1];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_125, getitem_197, getitem_198);  permute_125 = getitem_197 = getitem_198 = None
    getitem_199: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_11[0];  _scaled_dot_product_flash_attention_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_130: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_199, [0, 2, 1, 3]);  getitem_199 = None
    view_187: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_130, [8, 196, 320]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_188: "f32[1568, 320]" = torch.ops.aten.view.default(view_187, [1568, 320]);  view_187 = None
    permute_131: "f32[320, 320]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    addmm_57: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg227_1, view_188, permute_131);  arg227_1 = view_188 = permute_131 = None
    view_189: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_57, [8, 196, 320]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_41: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_112: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_107, clone_41);  add_107 = clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_38 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_208: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_209: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_113: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-06);  getitem_208 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_38: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_112, getitem_209);  getitem_209 = None
    mul_109: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
    mul_110: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_109, arg228_1);  mul_109 = arg228_1 = None
    add_114: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_110, arg229_1);  mul_110 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_190: "f32[1568, 320]" = torch.ops.aten.view.default(add_114, [1568, 320]);  add_114 = None
    permute_132: "f32[320, 1280]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_58: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg231_1, view_190, permute_132);  arg231_1 = view_190 = permute_132 = None
    view_191: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1280]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_111: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
    mul_112: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476);  view_191 = None
    erf_11: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_115: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_113: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_111, add_115);  mul_111 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_42: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_113);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_192: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_42, [1568, 1280]);  clone_42 = None
    permute_133: "f32[1280, 320]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
    addmm_59: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg233_1, view_192, permute_133);  arg233_1 = view_192 = permute_133 = None
    view_193: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_59, [8, 196, 320]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_43: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_193);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_116: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_112, clone_43);  add_112 = clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_39 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
    getitem_210: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_211: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_117: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_39: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_116, getitem_211);  getitem_211 = None
    mul_114: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
    mul_115: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_114, arg234_1);  mul_114 = arg234_1 = None
    add_118: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_115, arg235_1);  mul_115 = arg235_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_194: "f32[1568, 320]" = torch.ops.aten.view.default(add_118, [1568, 320])
    permute_134: "f32[320, 320]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    addmm_60: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg237_1, view_194, permute_134);  arg237_1 = view_194 = permute_134 = None
    view_195: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_60, [8, 196, 320]);  addmm_60 = None
    view_196: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_195, [8, 196, 5, 64]);  view_195 = None
    permute_135: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_136: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_118, [0, 2, 1]);  add_118 = None
    view_197: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_136, [8, 320, 14, 14]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_18: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_197, arg238_1, arg239_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_197 = arg238_1 = arg239_1 = None
    view_198: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_18, [8, 320, 49]);  convolution_18 = None
    permute_137: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_198, [0, 2, 1]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_40 = torch.ops.aten.var_mean.correction(permute_137, [2], correction = 0, keepdim = True)
    getitem_212: "f32[8, 49, 1]" = var_mean_40[0]
    getitem_213: "f32[8, 49, 1]" = var_mean_40[1];  var_mean_40 = None
    add_119: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
    rsqrt_40: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_40: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_137, getitem_213);  permute_137 = getitem_213 = None
    mul_116: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
    mul_117: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_116, arg240_1);  mul_116 = arg240_1 = None
    add_120: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_117, arg241_1);  mul_117 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_199: "f32[392, 320]" = torch.ops.aten.view.default(add_120, [392, 320]);  add_120 = None
    permute_138: "f32[320, 640]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
    addmm_61: "f32[392, 640]" = torch.ops.aten.addmm.default(arg243_1, view_199, permute_138);  arg243_1 = view_199 = permute_138 = None
    view_200: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_61, [8, 49, 640]);  addmm_61 = None
    view_201: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_200, [8, -1, 2, 5, 64]);  view_200 = None
    permute_139: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_139);  permute_139 = None
    getitem_214: "f32[8, 5, 49, 64]" = unbind_12[0]
    getitem_215: "f32[8, 5, 49, 64]" = unbind_12[1];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_12 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_135, getitem_214, getitem_215);  permute_135 = getitem_214 = getitem_215 = None
    getitem_216: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_12[0];  _scaled_dot_product_flash_attention_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_140: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
    view_202: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_140, [8, 196, 320]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_203: "f32[1568, 320]" = torch.ops.aten.view.default(view_202, [1568, 320]);  view_202 = None
    permute_141: "f32[320, 320]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    addmm_62: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg245_1, view_203, permute_141);  arg245_1 = view_203 = permute_141 = None
    view_204: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_62, [8, 196, 320]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_44: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_204);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_121: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_116, clone_44);  add_116 = clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_41 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_225: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_226: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_122: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_225, 1e-06);  getitem_225 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_41: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_121, getitem_226);  getitem_226 = None
    mul_118: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
    mul_119: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_118, arg246_1);  mul_118 = arg246_1 = None
    add_123: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_119, arg247_1);  mul_119 = arg247_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_205: "f32[1568, 320]" = torch.ops.aten.view.default(add_123, [1568, 320]);  add_123 = None
    permute_142: "f32[320, 1280]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    addmm_63: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg249_1, view_205, permute_142);  arg249_1 = view_205 = permute_142 = None
    view_206: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_63, [8, 196, 1280]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_120: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
    mul_121: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
    erf_12: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
    add_124: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_122: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_120, add_124);  mul_120 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_45: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_122);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_207: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_45, [1568, 1280]);  clone_45 = None
    permute_143: "f32[1280, 320]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_64: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg251_1, view_207, permute_143);  arg251_1 = view_207 = permute_143 = None
    view_208: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_64, [8, 196, 320]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_46: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_208);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_125: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_121, clone_46);  add_121 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_42 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
    getitem_227: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_228: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_227, 1e-06);  getitem_227 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_42: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_125, getitem_228);  getitem_228 = None
    mul_123: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
    mul_124: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_123, arg252_1);  mul_123 = arg252_1 = None
    add_127: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_124, arg253_1);  mul_124 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_209: "f32[1568, 320]" = torch.ops.aten.view.default(add_127, [1568, 320])
    permute_144: "f32[320, 320]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_65: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg255_1, view_209, permute_144);  arg255_1 = view_209 = permute_144 = None
    view_210: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_65, [8, 196, 320]);  addmm_65 = None
    view_211: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_210, [8, 196, 5, 64]);  view_210 = None
    permute_145: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_146: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_127, [0, 2, 1]);  add_127 = None
    view_212: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_146, [8, 320, 14, 14]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_19: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_212, arg256_1, arg257_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_212 = arg256_1 = arg257_1 = None
    view_213: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_19, [8, 320, 49]);  convolution_19 = None
    permute_147: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_43 = torch.ops.aten.var_mean.correction(permute_147, [2], correction = 0, keepdim = True)
    getitem_229: "f32[8, 49, 1]" = var_mean_43[0]
    getitem_230: "f32[8, 49, 1]" = var_mean_43[1];  var_mean_43 = None
    add_128: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_229, 1e-05);  getitem_229 = None
    rsqrt_43: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_43: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_147, getitem_230);  permute_147 = getitem_230 = None
    mul_125: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
    mul_126: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_125, arg258_1);  mul_125 = arg258_1 = None
    add_129: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_126, arg259_1);  mul_126 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_214: "f32[392, 320]" = torch.ops.aten.view.default(add_129, [392, 320]);  add_129 = None
    permute_148: "f32[320, 640]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm_66: "f32[392, 640]" = torch.ops.aten.addmm.default(arg261_1, view_214, permute_148);  arg261_1 = view_214 = permute_148 = None
    view_215: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_66, [8, 49, 640]);  addmm_66 = None
    view_216: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_215, [8, -1, 2, 5, 64]);  view_215 = None
    permute_149: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_216, [2, 0, 3, 1, 4]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_13 = torch.ops.aten.unbind.int(permute_149);  permute_149 = None
    getitem_231: "f32[8, 5, 49, 64]" = unbind_13[0]
    getitem_232: "f32[8, 5, 49, 64]" = unbind_13[1];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_13 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_145, getitem_231, getitem_232);  permute_145 = getitem_231 = getitem_232 = None
    getitem_233: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_13[0];  _scaled_dot_product_flash_attention_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_150: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_233, [0, 2, 1, 3]);  getitem_233 = None
    view_217: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_150, [8, 196, 320]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_218: "f32[1568, 320]" = torch.ops.aten.view.default(view_217, [1568, 320]);  view_217 = None
    permute_151: "f32[320, 320]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
    addmm_67: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg263_1, view_218, permute_151);  arg263_1 = view_218 = permute_151 = None
    view_219: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_67, [8, 196, 320]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_47: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_219);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_130: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_125, clone_47);  add_125 = clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_44 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
    getitem_242: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_243: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_131: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_44: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_130, getitem_243);  getitem_243 = None
    mul_127: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
    mul_128: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_127, arg264_1);  mul_127 = arg264_1 = None
    add_132: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_128, arg265_1);  mul_128 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_220: "f32[1568, 320]" = torch.ops.aten.view.default(add_132, [1568, 320]);  add_132 = None
    permute_152: "f32[320, 1280]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    addmm_68: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg267_1, view_220, permute_152);  arg267_1 = view_220 = permute_152 = None
    view_221: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_68, [8, 196, 1280]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_129: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, 0.5)
    mul_130: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, 0.7071067811865476);  view_221 = None
    erf_13: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
    add_133: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_131: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_129, add_133);  mul_129 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_48: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_131);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_222: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_48, [1568, 1280]);  clone_48 = None
    permute_153: "f32[1280, 320]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
    addmm_69: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg269_1, view_222, permute_153);  arg269_1 = view_222 = permute_153 = None
    view_223: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_69, [8, 196, 320]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_49: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_223);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_134: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_130, clone_49);  add_130 = clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_45 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
    getitem_244: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_245: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_135: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_45: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_134, getitem_245);  getitem_245 = None
    mul_132: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
    mul_133: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_132, arg270_1);  mul_132 = arg270_1 = None
    add_136: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_133, arg271_1);  mul_133 = arg271_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_224: "f32[1568, 320]" = torch.ops.aten.view.default(add_136, [1568, 320])
    permute_154: "f32[320, 320]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
    addmm_70: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg273_1, view_224, permute_154);  arg273_1 = view_224 = permute_154 = None
    view_225: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_70, [8, 196, 320]);  addmm_70 = None
    view_226: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_225, [8, 196, 5, 64]);  view_225 = None
    permute_155: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_156: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None
    view_227: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_156, [8, 320, 14, 14]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_20: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_227, arg274_1, arg275_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_227 = arg274_1 = arg275_1 = None
    view_228: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_20, [8, 320, 49]);  convolution_20 = None
    permute_157: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_46 = torch.ops.aten.var_mean.correction(permute_157, [2], correction = 0, keepdim = True)
    getitem_246: "f32[8, 49, 1]" = var_mean_46[0]
    getitem_247: "f32[8, 49, 1]" = var_mean_46[1];  var_mean_46 = None
    add_137: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
    rsqrt_46: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_46: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_157, getitem_247);  permute_157 = getitem_247 = None
    mul_134: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
    mul_135: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_134, arg276_1);  mul_134 = arg276_1 = None
    add_138: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_135, arg277_1);  mul_135 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_229: "f32[392, 320]" = torch.ops.aten.view.default(add_138, [392, 320]);  add_138 = None
    permute_158: "f32[320, 640]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    addmm_71: "f32[392, 640]" = torch.ops.aten.addmm.default(arg279_1, view_229, permute_158);  arg279_1 = view_229 = permute_158 = None
    view_230: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_71, [8, 49, 640]);  addmm_71 = None
    view_231: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_230, [8, -1, 2, 5, 64]);  view_230 = None
    permute_159: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_231, [2, 0, 3, 1, 4]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_14 = torch.ops.aten.unbind.int(permute_159);  permute_159 = None
    getitem_248: "f32[8, 5, 49, 64]" = unbind_14[0]
    getitem_249: "f32[8, 5, 49, 64]" = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_14 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_155, getitem_248, getitem_249);  permute_155 = getitem_248 = getitem_249 = None
    getitem_250: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_14[0];  _scaled_dot_product_flash_attention_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_160: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_250, [0, 2, 1, 3]);  getitem_250 = None
    view_232: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_160, [8, 196, 320]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_233: "f32[1568, 320]" = torch.ops.aten.view.default(view_232, [1568, 320]);  view_232 = None
    permute_161: "f32[320, 320]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
    addmm_72: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg281_1, view_233, permute_161);  arg281_1 = view_233 = permute_161 = None
    view_234: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_72, [8, 196, 320]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_50: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_234);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_139: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_134, clone_50);  add_134 = clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_47 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_259: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_260: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_259, 1e-06);  getitem_259 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_47: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_139, getitem_260);  getitem_260 = None
    mul_136: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
    mul_137: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_136, arg282_1);  mul_136 = arg282_1 = None
    add_141: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_137, arg283_1);  mul_137 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_235: "f32[1568, 320]" = torch.ops.aten.view.default(add_141, [1568, 320]);  add_141 = None
    permute_162: "f32[320, 1280]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
    addmm_73: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg285_1, view_235, permute_162);  arg285_1 = view_235 = permute_162 = None
    view_236: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_73, [8, 196, 1280]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_138: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, 0.5)
    mul_139: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, 0.7071067811865476);  view_236 = None
    erf_14: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_142: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_140: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_138, add_142);  mul_138 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_51: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_237: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_51, [1568, 1280]);  clone_51 = None
    permute_163: "f32[1280, 320]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
    addmm_74: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg287_1, view_237, permute_163);  arg287_1 = view_237 = permute_163 = None
    view_238: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_74, [8, 196, 320]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_52: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_238);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_143: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_139, clone_52);  add_139 = clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_48 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
    getitem_261: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_262: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_144: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_261, 1e-06);  getitem_261 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_48: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_143, getitem_262);  getitem_262 = None
    mul_141: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
    mul_142: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_141, arg288_1);  mul_141 = arg288_1 = None
    add_145: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_142, arg289_1);  mul_142 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_239: "f32[1568, 320]" = torch.ops.aten.view.default(add_145, [1568, 320])
    permute_164: "f32[320, 320]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
    addmm_75: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg291_1, view_239, permute_164);  arg291_1 = view_239 = permute_164 = None
    view_240: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_75, [8, 196, 320]);  addmm_75 = None
    view_241: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_240, [8, 196, 5, 64]);  view_240 = None
    permute_165: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_166: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_145, [0, 2, 1]);  add_145 = None
    view_242: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_166, [8, 320, 14, 14]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_21: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_242, arg292_1, arg293_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_242 = arg292_1 = arg293_1 = None
    view_243: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_21, [8, 320, 49]);  convolution_21 = None
    permute_167: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_49 = torch.ops.aten.var_mean.correction(permute_167, [2], correction = 0, keepdim = True)
    getitem_263: "f32[8, 49, 1]" = var_mean_49[0]
    getitem_264: "f32[8, 49, 1]" = var_mean_49[1];  var_mean_49 = None
    add_146: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_263, 1e-05);  getitem_263 = None
    rsqrt_49: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_49: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_167, getitem_264);  permute_167 = getitem_264 = None
    mul_143: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
    mul_144: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_143, arg294_1);  mul_143 = arg294_1 = None
    add_147: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_144, arg295_1);  mul_144 = arg295_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_244: "f32[392, 320]" = torch.ops.aten.view.default(add_147, [392, 320]);  add_147 = None
    permute_168: "f32[320, 640]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
    addmm_76: "f32[392, 640]" = torch.ops.aten.addmm.default(arg297_1, view_244, permute_168);  arg297_1 = view_244 = permute_168 = None
    view_245: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_76, [8, 49, 640]);  addmm_76 = None
    view_246: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_245, [8, -1, 2, 5, 64]);  view_245 = None
    permute_169: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_246, [2, 0, 3, 1, 4]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_15 = torch.ops.aten.unbind.int(permute_169);  permute_169 = None
    getitem_265: "f32[8, 5, 49, 64]" = unbind_15[0]
    getitem_266: "f32[8, 5, 49, 64]" = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_15 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_165, getitem_265, getitem_266);  permute_165 = getitem_265 = getitem_266 = None
    getitem_267: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_15[0];  _scaled_dot_product_flash_attention_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_170: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_267, [0, 2, 1, 3]);  getitem_267 = None
    view_247: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_170, [8, 196, 320]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_248: "f32[1568, 320]" = torch.ops.aten.view.default(view_247, [1568, 320]);  view_247 = None
    permute_171: "f32[320, 320]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
    addmm_77: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg299_1, view_248, permute_171);  arg299_1 = view_248 = permute_171 = None
    view_249: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_77, [8, 196, 320]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_53: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_249);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_148: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_143, clone_53);  add_143 = clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_50 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
    getitem_276: "f32[8, 196, 1]" = var_mean_50[0]
    getitem_277: "f32[8, 196, 1]" = var_mean_50[1];  var_mean_50 = None
    add_149: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-06);  getitem_276 = None
    rsqrt_50: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_50: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_148, getitem_277);  getitem_277 = None
    mul_145: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
    mul_146: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_145, arg300_1);  mul_145 = arg300_1 = None
    add_150: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_146, arg301_1);  mul_146 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_250: "f32[1568, 320]" = torch.ops.aten.view.default(add_150, [1568, 320]);  add_150 = None
    permute_172: "f32[320, 1280]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
    addmm_78: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg303_1, view_250, permute_172);  arg303_1 = view_250 = permute_172 = None
    view_251: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_78, [8, 196, 1280]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_147: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    mul_148: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, 0.7071067811865476);  view_251 = None
    erf_15: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_151: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_149: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_147, add_151);  mul_147 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_149);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_252: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_54, [1568, 1280]);  clone_54 = None
    permute_173: "f32[1280, 320]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
    addmm_79: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg305_1, view_252, permute_173);  arg305_1 = view_252 = permute_173 = None
    view_253: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_79, [8, 196, 320]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_253);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_152: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_148, clone_55);  add_148 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_51 = torch.ops.aten.var_mean.correction(add_152, [2], correction = 0, keepdim = True)
    getitem_278: "f32[8, 196, 1]" = var_mean_51[0]
    getitem_279: "f32[8, 196, 1]" = var_mean_51[1];  var_mean_51 = None
    add_153: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
    rsqrt_51: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    sub_51: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_152, getitem_279);  getitem_279 = None
    mul_150: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
    mul_151: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_150, arg306_1);  mul_150 = arg306_1 = None
    add_154: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_151, arg307_1);  mul_151 = arg307_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_254: "f32[1568, 320]" = torch.ops.aten.view.default(add_154, [1568, 320])
    permute_174: "f32[320, 320]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
    addmm_80: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg309_1, view_254, permute_174);  arg309_1 = view_254 = permute_174 = None
    view_255: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_80, [8, 196, 320]);  addmm_80 = None
    view_256: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_255, [8, 196, 5, 64]);  view_255 = None
    permute_175: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_176: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_154, [0, 2, 1]);  add_154 = None
    view_257: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_176, [8, 320, 14, 14]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_22: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_257, arg310_1, arg311_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_257 = arg310_1 = arg311_1 = None
    view_258: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_22, [8, 320, 49]);  convolution_22 = None
    permute_177: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_52 = torch.ops.aten.var_mean.correction(permute_177, [2], correction = 0, keepdim = True)
    getitem_280: "f32[8, 49, 1]" = var_mean_52[0]
    getitem_281: "f32[8, 49, 1]" = var_mean_52[1];  var_mean_52 = None
    add_155: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-05);  getitem_280 = None
    rsqrt_52: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_52: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_177, getitem_281);  permute_177 = getitem_281 = None
    mul_152: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
    mul_153: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_152, arg312_1);  mul_152 = arg312_1 = None
    add_156: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_153, arg313_1);  mul_153 = arg313_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_259: "f32[392, 320]" = torch.ops.aten.view.default(add_156, [392, 320]);  add_156 = None
    permute_178: "f32[320, 640]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
    addmm_81: "f32[392, 640]" = torch.ops.aten.addmm.default(arg315_1, view_259, permute_178);  arg315_1 = view_259 = permute_178 = None
    view_260: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_81, [8, 49, 640]);  addmm_81 = None
    view_261: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_260, [8, -1, 2, 5, 64]);  view_260 = None
    permute_179: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_261, [2, 0, 3, 1, 4]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_16 = torch.ops.aten.unbind.int(permute_179);  permute_179 = None
    getitem_282: "f32[8, 5, 49, 64]" = unbind_16[0]
    getitem_283: "f32[8, 5, 49, 64]" = unbind_16[1];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_16 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_175, getitem_282, getitem_283);  permute_175 = getitem_282 = getitem_283 = None
    getitem_284: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_16[0];  _scaled_dot_product_flash_attention_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_180: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_284, [0, 2, 1, 3]);  getitem_284 = None
    view_262: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_180, [8, 196, 320]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_263: "f32[1568, 320]" = torch.ops.aten.view.default(view_262, [1568, 320]);  view_262 = None
    permute_181: "f32[320, 320]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
    addmm_82: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg317_1, view_263, permute_181);  arg317_1 = view_263 = permute_181 = None
    view_264: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_82, [8, 196, 320]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_56: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_264);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_157: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_152, clone_56);  add_152 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_53 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_293: "f32[8, 196, 1]" = var_mean_53[0]
    getitem_294: "f32[8, 196, 1]" = var_mean_53[1];  var_mean_53 = None
    add_158: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_293, 1e-06);  getitem_293 = None
    rsqrt_53: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_53: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_157, getitem_294);  getitem_294 = None
    mul_154: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
    mul_155: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_154, arg318_1);  mul_154 = arg318_1 = None
    add_159: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_155, arg319_1);  mul_155 = arg319_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_265: "f32[1568, 320]" = torch.ops.aten.view.default(add_159, [1568, 320]);  add_159 = None
    permute_182: "f32[320, 1280]" = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
    addmm_83: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg321_1, view_265, permute_182);  arg321_1 = view_265 = permute_182 = None
    view_266: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_83, [8, 196, 1280]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_156: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, 0.5)
    mul_157: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, 0.7071067811865476);  view_266 = None
    erf_16: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_157);  mul_157 = None
    add_160: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_158: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_156, add_160);  mul_156 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_57: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_158);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_267: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_57, [1568, 1280]);  clone_57 = None
    permute_183: "f32[1280, 320]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
    addmm_84: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg323_1, view_267, permute_183);  arg323_1 = view_267 = permute_183 = None
    view_268: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_84, [8, 196, 320]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_58: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_268);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_161: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_157, clone_58);  add_157 = clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_54 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
    getitem_295: "f32[8, 196, 1]" = var_mean_54[0]
    getitem_296: "f32[8, 196, 1]" = var_mean_54[1];  var_mean_54 = None
    add_162: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_295, 1e-06);  getitem_295 = None
    rsqrt_54: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_54: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_161, getitem_296);  getitem_296 = None
    mul_159: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
    mul_160: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_159, arg324_1);  mul_159 = arg324_1 = None
    add_163: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_160, arg325_1);  mul_160 = arg325_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_269: "f32[1568, 320]" = torch.ops.aten.view.default(add_163, [1568, 320])
    permute_184: "f32[320, 320]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
    addmm_85: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg327_1, view_269, permute_184);  arg327_1 = view_269 = permute_184 = None
    view_270: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_85, [8, 196, 320]);  addmm_85 = None
    view_271: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_270, [8, 196, 5, 64]);  view_270 = None
    permute_185: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_186: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_163, [0, 2, 1]);  add_163 = None
    view_272: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_186, [8, 320, 14, 14]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_23: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_272, arg328_1, arg329_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_272 = arg328_1 = arg329_1 = None
    view_273: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_23, [8, 320, 49]);  convolution_23 = None
    permute_187: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_55 = torch.ops.aten.var_mean.correction(permute_187, [2], correction = 0, keepdim = True)
    getitem_297: "f32[8, 49, 1]" = var_mean_55[0]
    getitem_298: "f32[8, 49, 1]" = var_mean_55[1];  var_mean_55 = None
    add_164: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_297, 1e-05);  getitem_297 = None
    rsqrt_55: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_55: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_187, getitem_298);  permute_187 = getitem_298 = None
    mul_161: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
    mul_162: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_161, arg330_1);  mul_161 = arg330_1 = None
    add_165: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_162, arg331_1);  mul_162 = arg331_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_274: "f32[392, 320]" = torch.ops.aten.view.default(add_165, [392, 320]);  add_165 = None
    permute_188: "f32[320, 640]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
    addmm_86: "f32[392, 640]" = torch.ops.aten.addmm.default(arg333_1, view_274, permute_188);  arg333_1 = view_274 = permute_188 = None
    view_275: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_86, [8, 49, 640]);  addmm_86 = None
    view_276: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_275, [8, -1, 2, 5, 64]);  view_275 = None
    permute_189: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_276, [2, 0, 3, 1, 4]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_17 = torch.ops.aten.unbind.int(permute_189);  permute_189 = None
    getitem_299: "f32[8, 5, 49, 64]" = unbind_17[0]
    getitem_300: "f32[8, 5, 49, 64]" = unbind_17[1];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_17 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_185, getitem_299, getitem_300);  permute_185 = getitem_299 = getitem_300 = None
    getitem_301: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_17[0];  _scaled_dot_product_flash_attention_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_190: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_301, [0, 2, 1, 3]);  getitem_301 = None
    view_277: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_190, [8, 196, 320]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_278: "f32[1568, 320]" = torch.ops.aten.view.default(view_277, [1568, 320]);  view_277 = None
    permute_191: "f32[320, 320]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
    addmm_87: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg335_1, view_278, permute_191);  arg335_1 = view_278 = permute_191 = None
    view_279: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_87, [8, 196, 320]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_59: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_279);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_166: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_161, clone_59);  add_161 = clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_56 = torch.ops.aten.var_mean.correction(add_166, [2], correction = 0, keepdim = True)
    getitem_310: "f32[8, 196, 1]" = var_mean_56[0]
    getitem_311: "f32[8, 196, 1]" = var_mean_56[1];  var_mean_56 = None
    add_167: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-06);  getitem_310 = None
    rsqrt_56: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    sub_56: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_166, getitem_311);  getitem_311 = None
    mul_163: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
    mul_164: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_163, arg336_1);  mul_163 = arg336_1 = None
    add_168: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_164, arg337_1);  mul_164 = arg337_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_280: "f32[1568, 320]" = torch.ops.aten.view.default(add_168, [1568, 320]);  add_168 = None
    permute_192: "f32[320, 1280]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
    addmm_88: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg339_1, view_280, permute_192);  arg339_1 = view_280 = permute_192 = None
    view_281: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_88, [8, 196, 1280]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_165: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, 0.5)
    mul_166: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, 0.7071067811865476);  view_281 = None
    erf_17: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
    add_169: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_167: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_165, add_169);  mul_165 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_60: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_167);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_282: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_60, [1568, 1280]);  clone_60 = None
    permute_193: "f32[1280, 320]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
    addmm_89: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg341_1, view_282, permute_193);  arg341_1 = view_282 = permute_193 = None
    view_283: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_89, [8, 196, 320]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_61: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_283);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_170: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_166, clone_61);  add_166 = clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_57 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
    getitem_312: "f32[8, 196, 1]" = var_mean_57[0]
    getitem_313: "f32[8, 196, 1]" = var_mean_57[1];  var_mean_57 = None
    add_171: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-06);  getitem_312 = None
    rsqrt_57: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_57: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_170, getitem_313);  getitem_313 = None
    mul_168: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
    mul_169: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_168, arg342_1);  mul_168 = arg342_1 = None
    add_172: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_169, arg343_1);  mul_169 = arg343_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_284: "f32[1568, 320]" = torch.ops.aten.view.default(add_172, [1568, 320])
    permute_194: "f32[320, 320]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
    addmm_90: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg345_1, view_284, permute_194);  arg345_1 = view_284 = permute_194 = None
    view_285: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_90, [8, 196, 320]);  addmm_90 = None
    view_286: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_285, [8, 196, 5, 64]);  view_285 = None
    permute_195: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_196: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
    view_287: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_196, [8, 320, 14, 14]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_24: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_287, arg346_1, arg347_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_287 = arg346_1 = arg347_1 = None
    view_288: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_24, [8, 320, 49]);  convolution_24 = None
    permute_197: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_58 = torch.ops.aten.var_mean.correction(permute_197, [2], correction = 0, keepdim = True)
    getitem_314: "f32[8, 49, 1]" = var_mean_58[0]
    getitem_315: "f32[8, 49, 1]" = var_mean_58[1];  var_mean_58 = None
    add_173: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_314, 1e-05);  getitem_314 = None
    rsqrt_58: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    sub_58: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_197, getitem_315);  permute_197 = getitem_315 = None
    mul_170: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
    mul_171: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_170, arg348_1);  mul_170 = arg348_1 = None
    add_174: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_171, arg349_1);  mul_171 = arg349_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_289: "f32[392, 320]" = torch.ops.aten.view.default(add_174, [392, 320]);  add_174 = None
    permute_198: "f32[320, 640]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
    addmm_91: "f32[392, 640]" = torch.ops.aten.addmm.default(arg351_1, view_289, permute_198);  arg351_1 = view_289 = permute_198 = None
    view_290: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_91, [8, 49, 640]);  addmm_91 = None
    view_291: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_290, [8, -1, 2, 5, 64]);  view_290 = None
    permute_199: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_291, [2, 0, 3, 1, 4]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_18 = torch.ops.aten.unbind.int(permute_199);  permute_199 = None
    getitem_316: "f32[8, 5, 49, 64]" = unbind_18[0]
    getitem_317: "f32[8, 5, 49, 64]" = unbind_18[1];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_18 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_195, getitem_316, getitem_317);  permute_195 = getitem_316 = getitem_317 = None
    getitem_318: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_18[0];  _scaled_dot_product_flash_attention_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_200: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_318, [0, 2, 1, 3]);  getitem_318 = None
    view_292: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_200, [8, 196, 320]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_293: "f32[1568, 320]" = torch.ops.aten.view.default(view_292, [1568, 320]);  view_292 = None
    permute_201: "f32[320, 320]" = torch.ops.aten.permute.default(arg352_1, [1, 0]);  arg352_1 = None
    addmm_92: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg353_1, view_293, permute_201);  arg353_1 = view_293 = permute_201 = None
    view_294: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_92, [8, 196, 320]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_62: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_294);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_175: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_170, clone_62);  add_170 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_59 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
    getitem_327: "f32[8, 196, 1]" = var_mean_59[0]
    getitem_328: "f32[8, 196, 1]" = var_mean_59[1];  var_mean_59 = None
    add_176: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_327, 1e-06);  getitem_327 = None
    rsqrt_59: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_59: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_175, getitem_328);  getitem_328 = None
    mul_172: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
    mul_173: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_172, arg354_1);  mul_172 = arg354_1 = None
    add_177: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_173, arg355_1);  mul_173 = arg355_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_295: "f32[1568, 320]" = torch.ops.aten.view.default(add_177, [1568, 320]);  add_177 = None
    permute_202: "f32[320, 1280]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
    addmm_93: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg357_1, view_295, permute_202);  arg357_1 = view_295 = permute_202 = None
    view_296: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_93, [8, 196, 1280]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_174: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, 0.5)
    mul_175: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, 0.7071067811865476);  view_296 = None
    erf_18: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_178: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_176: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_174, add_178);  mul_174 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_176);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_297: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_63, [1568, 1280]);  clone_63 = None
    permute_203: "f32[1280, 320]" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
    addmm_94: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg359_1, view_297, permute_203);  arg359_1 = view_297 = permute_203 = None
    view_298: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_94, [8, 196, 320]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_298);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_179: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_175, clone_64);  add_175 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_60 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
    getitem_329: "f32[8, 196, 1]" = var_mean_60[0]
    getitem_330: "f32[8, 196, 1]" = var_mean_60[1];  var_mean_60 = None
    add_180: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_329, 1e-06);  getitem_329 = None
    rsqrt_60: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_60: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_179, getitem_330);  getitem_330 = None
    mul_177: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
    mul_178: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_177, arg360_1);  mul_177 = arg360_1 = None
    add_181: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_178, arg361_1);  mul_178 = arg361_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_299: "f32[1568, 320]" = torch.ops.aten.view.default(add_181, [1568, 320])
    permute_204: "f32[320, 320]" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
    addmm_95: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg363_1, view_299, permute_204);  arg363_1 = view_299 = permute_204 = None
    view_300: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_95, [8, 196, 320]);  addmm_95 = None
    view_301: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_300, [8, 196, 5, 64]);  view_300 = None
    permute_205: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_206: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_181, [0, 2, 1]);  add_181 = None
    view_302: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_206, [8, 320, 14, 14]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_25: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_302, arg364_1, arg365_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_302 = arg364_1 = arg365_1 = None
    view_303: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_25, [8, 320, 49]);  convolution_25 = None
    permute_207: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_61 = torch.ops.aten.var_mean.correction(permute_207, [2], correction = 0, keepdim = True)
    getitem_331: "f32[8, 49, 1]" = var_mean_61[0]
    getitem_332: "f32[8, 49, 1]" = var_mean_61[1];  var_mean_61 = None
    add_182: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_331, 1e-05);  getitem_331 = None
    rsqrt_61: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_61: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_207, getitem_332);  permute_207 = getitem_332 = None
    mul_179: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
    mul_180: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_179, arg366_1);  mul_179 = arg366_1 = None
    add_183: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_180, arg367_1);  mul_180 = arg367_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_304: "f32[392, 320]" = torch.ops.aten.view.default(add_183, [392, 320]);  add_183 = None
    permute_208: "f32[320, 640]" = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
    addmm_96: "f32[392, 640]" = torch.ops.aten.addmm.default(arg369_1, view_304, permute_208);  arg369_1 = view_304 = permute_208 = None
    view_305: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_96, [8, 49, 640]);  addmm_96 = None
    view_306: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_305, [8, -1, 2, 5, 64]);  view_305 = None
    permute_209: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_306, [2, 0, 3, 1, 4]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_19 = torch.ops.aten.unbind.int(permute_209);  permute_209 = None
    getitem_333: "f32[8, 5, 49, 64]" = unbind_19[0]
    getitem_334: "f32[8, 5, 49, 64]" = unbind_19[1];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_19 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_205, getitem_333, getitem_334);  permute_205 = getitem_333 = getitem_334 = None
    getitem_335: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_19[0];  _scaled_dot_product_flash_attention_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_210: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_335, [0, 2, 1, 3]);  getitem_335 = None
    view_307: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_210, [8, 196, 320]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_308: "f32[1568, 320]" = torch.ops.aten.view.default(view_307, [1568, 320]);  view_307 = None
    permute_211: "f32[320, 320]" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
    addmm_97: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg371_1, view_308, permute_211);  arg371_1 = view_308 = permute_211 = None
    view_309: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_97, [8, 196, 320]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_65: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_184: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_179, clone_65);  add_179 = clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_62 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
    getitem_344: "f32[8, 196, 1]" = var_mean_62[0]
    getitem_345: "f32[8, 196, 1]" = var_mean_62[1];  var_mean_62 = None
    add_185: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_344, 1e-06);  getitem_344 = None
    rsqrt_62: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_62: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_184, getitem_345);  getitem_345 = None
    mul_181: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
    mul_182: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_181, arg372_1);  mul_181 = arg372_1 = None
    add_186: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_182, arg373_1);  mul_182 = arg373_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_310: "f32[1568, 320]" = torch.ops.aten.view.default(add_186, [1568, 320]);  add_186 = None
    permute_212: "f32[320, 1280]" = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
    addmm_98: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg375_1, view_310, permute_212);  arg375_1 = view_310 = permute_212 = None
    view_311: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_98, [8, 196, 1280]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_183: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, 0.5)
    mul_184: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476);  view_311 = None
    erf_19: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
    add_187: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_185: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_183, add_187);  mul_183 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_66: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_185);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_312: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_66, [1568, 1280]);  clone_66 = None
    permute_213: "f32[1280, 320]" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
    addmm_99: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg377_1, view_312, permute_213);  arg377_1 = view_312 = permute_213 = None
    view_313: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_99, [8, 196, 320]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_67: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_313);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_188: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_184, clone_67);  add_184 = clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_63 = torch.ops.aten.var_mean.correction(add_188, [2], correction = 0, keepdim = True)
    getitem_346: "f32[8, 196, 1]" = var_mean_63[0]
    getitem_347: "f32[8, 196, 1]" = var_mean_63[1];  var_mean_63 = None
    add_189: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_346, 1e-06);  getitem_346 = None
    rsqrt_63: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_63: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_188, getitem_347);  getitem_347 = None
    mul_186: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
    mul_187: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_186, arg378_1);  mul_186 = arg378_1 = None
    add_190: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_187, arg379_1);  mul_187 = arg379_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_314: "f32[1568, 320]" = torch.ops.aten.view.default(add_190, [1568, 320])
    permute_214: "f32[320, 320]" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
    addmm_100: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg381_1, view_314, permute_214);  arg381_1 = view_314 = permute_214 = None
    view_315: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_100, [8, 196, 320]);  addmm_100 = None
    view_316: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_315, [8, 196, 5, 64]);  view_315 = None
    permute_215: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_216: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_190, [0, 2, 1]);  add_190 = None
    view_317: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_216, [8, 320, 14, 14]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_26: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_317, arg382_1, arg383_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_317 = arg382_1 = arg383_1 = None
    view_318: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_26, [8, 320, 49]);  convolution_26 = None
    permute_217: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_64 = torch.ops.aten.var_mean.correction(permute_217, [2], correction = 0, keepdim = True)
    getitem_348: "f32[8, 49, 1]" = var_mean_64[0]
    getitem_349: "f32[8, 49, 1]" = var_mean_64[1];  var_mean_64 = None
    add_191: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_348, 1e-05);  getitem_348 = None
    rsqrt_64: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_64: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_217, getitem_349);  permute_217 = getitem_349 = None
    mul_188: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
    mul_189: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_188, arg384_1);  mul_188 = arg384_1 = None
    add_192: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_189, arg385_1);  mul_189 = arg385_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_319: "f32[392, 320]" = torch.ops.aten.view.default(add_192, [392, 320]);  add_192 = None
    permute_218: "f32[320, 640]" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
    addmm_101: "f32[392, 640]" = torch.ops.aten.addmm.default(arg387_1, view_319, permute_218);  arg387_1 = view_319 = permute_218 = None
    view_320: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_101, [8, 49, 640]);  addmm_101 = None
    view_321: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_320, [8, -1, 2, 5, 64]);  view_320 = None
    permute_219: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_321, [2, 0, 3, 1, 4]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_20 = torch.ops.aten.unbind.int(permute_219);  permute_219 = None
    getitem_350: "f32[8, 5, 49, 64]" = unbind_20[0]
    getitem_351: "f32[8, 5, 49, 64]" = unbind_20[1];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_20 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_215, getitem_350, getitem_351);  permute_215 = getitem_350 = getitem_351 = None
    getitem_352: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_20[0];  _scaled_dot_product_flash_attention_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_220: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_352, [0, 2, 1, 3]);  getitem_352 = None
    view_322: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_220, [8, 196, 320]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_323: "f32[1568, 320]" = torch.ops.aten.view.default(view_322, [1568, 320]);  view_322 = None
    permute_221: "f32[320, 320]" = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
    addmm_102: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg389_1, view_323, permute_221);  arg389_1 = view_323 = permute_221 = None
    view_324: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_102, [8, 196, 320]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_68: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_324);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_193: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_188, clone_68);  add_188 = clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_65 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
    getitem_361: "f32[8, 196, 1]" = var_mean_65[0]
    getitem_362: "f32[8, 196, 1]" = var_mean_65[1];  var_mean_65 = None
    add_194: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_361, 1e-06);  getitem_361 = None
    rsqrt_65: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_65: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_193, getitem_362);  getitem_362 = None
    mul_190: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
    mul_191: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_190, arg390_1);  mul_190 = arg390_1 = None
    add_195: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_191, arg391_1);  mul_191 = arg391_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_325: "f32[1568, 320]" = torch.ops.aten.view.default(add_195, [1568, 320]);  add_195 = None
    permute_222: "f32[320, 1280]" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
    addmm_103: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg393_1, view_325, permute_222);  arg393_1 = view_325 = permute_222 = None
    view_326: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_103, [8, 196, 1280]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_192: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, 0.5)
    mul_193: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, 0.7071067811865476);  view_326 = None
    erf_20: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_193);  mul_193 = None
    add_196: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_194: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_192, add_196);  mul_192 = add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_69: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_194);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_327: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_69, [1568, 1280]);  clone_69 = None
    permute_223: "f32[1280, 320]" = torch.ops.aten.permute.default(arg394_1, [1, 0]);  arg394_1 = None
    addmm_104: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg395_1, view_327, permute_223);  arg395_1 = view_327 = permute_223 = None
    view_328: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_104, [8, 196, 320]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_70: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_328);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_197: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_193, clone_70);  add_193 = clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_66 = torch.ops.aten.var_mean.correction(add_197, [2], correction = 0, keepdim = True)
    getitem_363: "f32[8, 196, 1]" = var_mean_66[0]
    getitem_364: "f32[8, 196, 1]" = var_mean_66[1];  var_mean_66 = None
    add_198: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_363, 1e-06);  getitem_363 = None
    rsqrt_66: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_66: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_197, getitem_364);  getitem_364 = None
    mul_195: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
    mul_196: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_195, arg396_1);  mul_195 = arg396_1 = None
    add_199: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_196, arg397_1);  mul_196 = arg397_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_329: "f32[1568, 320]" = torch.ops.aten.view.default(add_199, [1568, 320])
    permute_224: "f32[320, 320]" = torch.ops.aten.permute.default(arg398_1, [1, 0]);  arg398_1 = None
    addmm_105: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg399_1, view_329, permute_224);  arg399_1 = view_329 = permute_224 = None
    view_330: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_105, [8, 196, 320]);  addmm_105 = None
    view_331: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_330, [8, 196, 5, 64]);  view_330 = None
    permute_225: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_226: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_199, [0, 2, 1]);  add_199 = None
    view_332: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_226, [8, 320, 14, 14]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_27: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_332, arg400_1, arg401_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_332 = arg400_1 = arg401_1 = None
    view_333: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_27, [8, 320, 49]);  convolution_27 = None
    permute_227: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_67 = torch.ops.aten.var_mean.correction(permute_227, [2], correction = 0, keepdim = True)
    getitem_365: "f32[8, 49, 1]" = var_mean_67[0]
    getitem_366: "f32[8, 49, 1]" = var_mean_67[1];  var_mean_67 = None
    add_200: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_365, 1e-05);  getitem_365 = None
    rsqrt_67: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    sub_67: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_227, getitem_366);  permute_227 = getitem_366 = None
    mul_197: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
    mul_198: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_197, arg402_1);  mul_197 = arg402_1 = None
    add_201: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_198, arg403_1);  mul_198 = arg403_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_334: "f32[392, 320]" = torch.ops.aten.view.default(add_201, [392, 320]);  add_201 = None
    permute_228: "f32[320, 640]" = torch.ops.aten.permute.default(arg404_1, [1, 0]);  arg404_1 = None
    addmm_106: "f32[392, 640]" = torch.ops.aten.addmm.default(arg405_1, view_334, permute_228);  arg405_1 = view_334 = permute_228 = None
    view_335: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_106, [8, 49, 640]);  addmm_106 = None
    view_336: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_335, [8, -1, 2, 5, 64]);  view_335 = None
    permute_229: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_336, [2, 0, 3, 1, 4]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_21 = torch.ops.aten.unbind.int(permute_229);  permute_229 = None
    getitem_367: "f32[8, 5, 49, 64]" = unbind_21[0]
    getitem_368: "f32[8, 5, 49, 64]" = unbind_21[1];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_21 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_225, getitem_367, getitem_368);  permute_225 = getitem_367 = getitem_368 = None
    getitem_369: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_21[0];  _scaled_dot_product_flash_attention_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_230: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_369, [0, 2, 1, 3]);  getitem_369 = None
    view_337: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_230, [8, 196, 320]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_338: "f32[1568, 320]" = torch.ops.aten.view.default(view_337, [1568, 320]);  view_337 = None
    permute_231: "f32[320, 320]" = torch.ops.aten.permute.default(arg406_1, [1, 0]);  arg406_1 = None
    addmm_107: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg407_1, view_338, permute_231);  arg407_1 = view_338 = permute_231 = None
    view_339: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_107, [8, 196, 320]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_71: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_339);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_202: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_197, clone_71);  add_197 = clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_68 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
    getitem_378: "f32[8, 196, 1]" = var_mean_68[0]
    getitem_379: "f32[8, 196, 1]" = var_mean_68[1];  var_mean_68 = None
    add_203: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_378, 1e-06);  getitem_378 = None
    rsqrt_68: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_68: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_202, getitem_379);  getitem_379 = None
    mul_199: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
    mul_200: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_199, arg408_1);  mul_199 = arg408_1 = None
    add_204: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_200, arg409_1);  mul_200 = arg409_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_340: "f32[1568, 320]" = torch.ops.aten.view.default(add_204, [1568, 320]);  add_204 = None
    permute_232: "f32[320, 1280]" = torch.ops.aten.permute.default(arg410_1, [1, 0]);  arg410_1 = None
    addmm_108: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg411_1, view_340, permute_232);  arg411_1 = view_340 = permute_232 = None
    view_341: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_108, [8, 196, 1280]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_201: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, 0.5)
    mul_202: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, 0.7071067811865476);  view_341 = None
    erf_21: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_202);  mul_202 = None
    add_205: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_203: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_201, add_205);  mul_201 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_72: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_203);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_342: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_72, [1568, 1280]);  clone_72 = None
    permute_233: "f32[1280, 320]" = torch.ops.aten.permute.default(arg412_1, [1, 0]);  arg412_1 = None
    addmm_109: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg413_1, view_342, permute_233);  arg413_1 = view_342 = permute_233 = None
    view_343: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_109, [8, 196, 320]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_73: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_343);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_206: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_202, clone_73);  add_202 = clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_69 = torch.ops.aten.var_mean.correction(add_206, [2], correction = 0, keepdim = True)
    getitem_380: "f32[8, 196, 1]" = var_mean_69[0]
    getitem_381: "f32[8, 196, 1]" = var_mean_69[1];  var_mean_69 = None
    add_207: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_380, 1e-06);  getitem_380 = None
    rsqrt_69: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_69: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_206, getitem_381);  getitem_381 = None
    mul_204: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
    mul_205: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_204, arg414_1);  mul_204 = arg414_1 = None
    add_208: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_205, arg415_1);  mul_205 = arg415_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_344: "f32[1568, 320]" = torch.ops.aten.view.default(add_208, [1568, 320])
    permute_234: "f32[320, 320]" = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
    addmm_110: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg417_1, view_344, permute_234);  arg417_1 = view_344 = permute_234 = None
    view_345: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_110, [8, 196, 320]);  addmm_110 = None
    view_346: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_345, [8, 196, 5, 64]);  view_345 = None
    permute_235: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_236: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_208, [0, 2, 1]);  add_208 = None
    view_347: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_236, [8, 320, 14, 14]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_28: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_347, arg418_1, arg419_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_347 = arg418_1 = arg419_1 = None
    view_348: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_28, [8, 320, 49]);  convolution_28 = None
    permute_237: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_348, [0, 2, 1]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_70 = torch.ops.aten.var_mean.correction(permute_237, [2], correction = 0, keepdim = True)
    getitem_382: "f32[8, 49, 1]" = var_mean_70[0]
    getitem_383: "f32[8, 49, 1]" = var_mean_70[1];  var_mean_70 = None
    add_209: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_382, 1e-05);  getitem_382 = None
    rsqrt_70: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    sub_70: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_237, getitem_383);  permute_237 = getitem_383 = None
    mul_206: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
    mul_207: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_206, arg420_1);  mul_206 = arg420_1 = None
    add_210: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_207, arg421_1);  mul_207 = arg421_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_349: "f32[392, 320]" = torch.ops.aten.view.default(add_210, [392, 320]);  add_210 = None
    permute_238: "f32[320, 640]" = torch.ops.aten.permute.default(arg422_1, [1, 0]);  arg422_1 = None
    addmm_111: "f32[392, 640]" = torch.ops.aten.addmm.default(arg423_1, view_349, permute_238);  arg423_1 = view_349 = permute_238 = None
    view_350: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_111, [8, 49, 640]);  addmm_111 = None
    view_351: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_350, [8, -1, 2, 5, 64]);  view_350 = None
    permute_239: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_351, [2, 0, 3, 1, 4]);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_22 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
    getitem_384: "f32[8, 5, 49, 64]" = unbind_22[0]
    getitem_385: "f32[8, 5, 49, 64]" = unbind_22[1];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_22 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_235, getitem_384, getitem_385);  permute_235 = getitem_384 = getitem_385 = None
    getitem_386: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_22[0];  _scaled_dot_product_flash_attention_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_240: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_386, [0, 2, 1, 3]);  getitem_386 = None
    view_352: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_240, [8, 196, 320]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_353: "f32[1568, 320]" = torch.ops.aten.view.default(view_352, [1568, 320]);  view_352 = None
    permute_241: "f32[320, 320]" = torch.ops.aten.permute.default(arg424_1, [1, 0]);  arg424_1 = None
    addmm_112: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg425_1, view_353, permute_241);  arg425_1 = view_353 = permute_241 = None
    view_354: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_112, [8, 196, 320]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_74: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_354);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_211: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_206, clone_74);  add_206 = clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_71 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
    getitem_395: "f32[8, 196, 1]" = var_mean_71[0]
    getitem_396: "f32[8, 196, 1]" = var_mean_71[1];  var_mean_71 = None
    add_212: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_395, 1e-06);  getitem_395 = None
    rsqrt_71: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    sub_71: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_211, getitem_396);  getitem_396 = None
    mul_208: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
    mul_209: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_208, arg426_1);  mul_208 = arg426_1 = None
    add_213: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_209, arg427_1);  mul_209 = arg427_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_355: "f32[1568, 320]" = torch.ops.aten.view.default(add_213, [1568, 320]);  add_213 = None
    permute_242: "f32[320, 1280]" = torch.ops.aten.permute.default(arg428_1, [1, 0]);  arg428_1 = None
    addmm_113: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg429_1, view_355, permute_242);  arg429_1 = view_355 = permute_242 = None
    view_356: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_113, [8, 196, 1280]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_210: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, 0.5)
    mul_211: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, 0.7071067811865476);  view_356 = None
    erf_22: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_214: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_212: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_210, add_214);  mul_210 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_75: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_212);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_357: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_75, [1568, 1280]);  clone_75 = None
    permute_243: "f32[1280, 320]" = torch.ops.aten.permute.default(arg430_1, [1, 0]);  arg430_1 = None
    addmm_114: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg431_1, view_357, permute_243);  arg431_1 = view_357 = permute_243 = None
    view_358: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_114, [8, 196, 320]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_76: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_358);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_215: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_211, clone_76);  add_211 = clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_72 = torch.ops.aten.var_mean.correction(add_215, [2], correction = 0, keepdim = True)
    getitem_397: "f32[8, 196, 1]" = var_mean_72[0]
    getitem_398: "f32[8, 196, 1]" = var_mean_72[1];  var_mean_72 = None
    add_216: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_397, 1e-06);  getitem_397 = None
    rsqrt_72: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_72: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_215, getitem_398);  getitem_398 = None
    mul_213: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
    mul_214: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_213, arg432_1);  mul_213 = arg432_1 = None
    add_217: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_214, arg433_1);  mul_214 = arg433_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_359: "f32[1568, 320]" = torch.ops.aten.view.default(add_217, [1568, 320])
    permute_244: "f32[320, 320]" = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
    addmm_115: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg435_1, view_359, permute_244);  arg435_1 = view_359 = permute_244 = None
    view_360: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_115, [8, 196, 320]);  addmm_115 = None
    view_361: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_360, [8, 196, 5, 64]);  view_360 = None
    permute_245: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_246: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_217, [0, 2, 1]);  add_217 = None
    view_362: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_246, [8, 320, 14, 14]);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_29: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_362, arg436_1, arg437_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_362 = arg436_1 = arg437_1 = None
    view_363: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_29, [8, 320, 49]);  convolution_29 = None
    permute_247: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_73 = torch.ops.aten.var_mean.correction(permute_247, [2], correction = 0, keepdim = True)
    getitem_399: "f32[8, 49, 1]" = var_mean_73[0]
    getitem_400: "f32[8, 49, 1]" = var_mean_73[1];  var_mean_73 = None
    add_218: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_399, 1e-05);  getitem_399 = None
    rsqrt_73: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_73: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_247, getitem_400);  permute_247 = getitem_400 = None
    mul_215: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
    mul_216: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_215, arg438_1);  mul_215 = arg438_1 = None
    add_219: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_216, arg439_1);  mul_216 = arg439_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_364: "f32[392, 320]" = torch.ops.aten.view.default(add_219, [392, 320]);  add_219 = None
    permute_248: "f32[320, 640]" = torch.ops.aten.permute.default(arg440_1, [1, 0]);  arg440_1 = None
    addmm_116: "f32[392, 640]" = torch.ops.aten.addmm.default(arg441_1, view_364, permute_248);  arg441_1 = view_364 = permute_248 = None
    view_365: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_116, [8, 49, 640]);  addmm_116 = None
    view_366: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_365, [8, -1, 2, 5, 64]);  view_365 = None
    permute_249: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_366, [2, 0, 3, 1, 4]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_23 = torch.ops.aten.unbind.int(permute_249);  permute_249 = None
    getitem_401: "f32[8, 5, 49, 64]" = unbind_23[0]
    getitem_402: "f32[8, 5, 49, 64]" = unbind_23[1];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_23 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_245, getitem_401, getitem_402);  permute_245 = getitem_401 = getitem_402 = None
    getitem_403: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_23[0];  _scaled_dot_product_flash_attention_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_250: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_403, [0, 2, 1, 3]);  getitem_403 = None
    view_367: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_250, [8, 196, 320]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_368: "f32[1568, 320]" = torch.ops.aten.view.default(view_367, [1568, 320]);  view_367 = None
    permute_251: "f32[320, 320]" = torch.ops.aten.permute.default(arg442_1, [1, 0]);  arg442_1 = None
    addmm_117: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg443_1, view_368, permute_251);  arg443_1 = view_368 = permute_251 = None
    view_369: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_117, [8, 196, 320]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_77: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_369);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_220: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_215, clone_77);  add_215 = clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_74 = torch.ops.aten.var_mean.correction(add_220, [2], correction = 0, keepdim = True)
    getitem_412: "f32[8, 196, 1]" = var_mean_74[0]
    getitem_413: "f32[8, 196, 1]" = var_mean_74[1];  var_mean_74 = None
    add_221: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_412, 1e-06);  getitem_412 = None
    rsqrt_74: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_74: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_220, getitem_413);  getitem_413 = None
    mul_217: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
    mul_218: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_217, arg444_1);  mul_217 = arg444_1 = None
    add_222: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_218, arg445_1);  mul_218 = arg445_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_370: "f32[1568, 320]" = torch.ops.aten.view.default(add_222, [1568, 320]);  add_222 = None
    permute_252: "f32[320, 1280]" = torch.ops.aten.permute.default(arg446_1, [1, 0]);  arg446_1 = None
    addmm_118: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg447_1, view_370, permute_252);  arg447_1 = view_370 = permute_252 = None
    view_371: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_118, [8, 196, 1280]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_219: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
    mul_220: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476);  view_371 = None
    erf_23: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
    add_223: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_221: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_219, add_223);  mul_219 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_78: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_221);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_372: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_78, [1568, 1280]);  clone_78 = None
    permute_253: "f32[1280, 320]" = torch.ops.aten.permute.default(arg448_1, [1, 0]);  arg448_1 = None
    addmm_119: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg449_1, view_372, permute_253);  arg449_1 = view_372 = permute_253 = None
    view_373: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_119, [8, 196, 320]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_79: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_373);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_224: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_220, clone_79);  add_220 = clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_75 = torch.ops.aten.var_mean.correction(add_224, [2], correction = 0, keepdim = True)
    getitem_414: "f32[8, 196, 1]" = var_mean_75[0]
    getitem_415: "f32[8, 196, 1]" = var_mean_75[1];  var_mean_75 = None
    add_225: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_414, 1e-06);  getitem_414 = None
    rsqrt_75: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
    sub_75: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_224, getitem_415);  getitem_415 = None
    mul_222: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
    mul_223: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_222, arg450_1);  mul_222 = arg450_1 = None
    add_226: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_223, arg451_1);  mul_223 = arg451_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_374: "f32[1568, 320]" = torch.ops.aten.view.default(add_226, [1568, 320])
    permute_254: "f32[320, 320]" = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
    addmm_120: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg453_1, view_374, permute_254);  arg453_1 = view_374 = permute_254 = None
    view_375: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_120, [8, 196, 320]);  addmm_120 = None
    view_376: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_375, [8, 196, 5, 64]);  view_375 = None
    permute_255: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_256: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_226, [0, 2, 1]);  add_226 = None
    view_377: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_256, [8, 320, 14, 14]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_30: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_377, arg454_1, arg455_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_377 = arg454_1 = arg455_1 = None
    view_378: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_30, [8, 320, 49]);  convolution_30 = None
    permute_257: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    var_mean_76 = torch.ops.aten.var_mean.correction(permute_257, [2], correction = 0, keepdim = True)
    getitem_416: "f32[8, 49, 1]" = var_mean_76[0]
    getitem_417: "f32[8, 49, 1]" = var_mean_76[1];  var_mean_76 = None
    add_227: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_416, 1e-05);  getitem_416 = None
    rsqrt_76: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    sub_76: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_257, getitem_417);  permute_257 = getitem_417 = None
    mul_224: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
    mul_225: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_224, arg456_1);  mul_224 = arg456_1 = None
    add_228: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_225, arg457_1);  mul_225 = arg457_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_379: "f32[392, 320]" = torch.ops.aten.view.default(add_228, [392, 320]);  add_228 = None
    permute_258: "f32[320, 640]" = torch.ops.aten.permute.default(arg458_1, [1, 0]);  arg458_1 = None
    addmm_121: "f32[392, 640]" = torch.ops.aten.addmm.default(arg459_1, view_379, permute_258);  arg459_1 = view_379 = permute_258 = None
    view_380: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_121, [8, 49, 640]);  addmm_121 = None
    view_381: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_380, [8, -1, 2, 5, 64]);  view_380 = None
    permute_259: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_381, [2, 0, 3, 1, 4]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_24 = torch.ops.aten.unbind.int(permute_259);  permute_259 = None
    getitem_418: "f32[8, 5, 49, 64]" = unbind_24[0]
    getitem_419: "f32[8, 5, 49, 64]" = unbind_24[1];  unbind_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_24 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_255, getitem_418, getitem_419);  permute_255 = getitem_418 = getitem_419 = None
    getitem_420: "f32[8, 5, 196, 64]" = _scaled_dot_product_flash_attention_24[0];  _scaled_dot_product_flash_attention_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_260: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_420, [0, 2, 1, 3]);  getitem_420 = None
    view_382: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_260, [8, 196, 320]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_383: "f32[1568, 320]" = torch.ops.aten.view.default(view_382, [1568, 320]);  view_382 = None
    permute_261: "f32[320, 320]" = torch.ops.aten.permute.default(arg460_1, [1, 0]);  arg460_1 = None
    addmm_122: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg461_1, view_383, permute_261);  arg461_1 = view_383 = permute_261 = None
    view_384: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_122, [8, 196, 320]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_80: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_384);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_229: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_224, clone_80);  add_224 = clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_77 = torch.ops.aten.var_mean.correction(add_229, [2], correction = 0, keepdim = True)
    getitem_429: "f32[8, 196, 1]" = var_mean_77[0]
    getitem_430: "f32[8, 196, 1]" = var_mean_77[1];  var_mean_77 = None
    add_230: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_429, 1e-06);  getitem_429 = None
    rsqrt_77: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    sub_77: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_229, getitem_430);  getitem_430 = None
    mul_226: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
    mul_227: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_226, arg462_1);  mul_226 = arg462_1 = None
    add_231: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_227, arg463_1);  mul_227 = arg463_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_385: "f32[1568, 320]" = torch.ops.aten.view.default(add_231, [1568, 320]);  add_231 = None
    permute_262: "f32[320, 1280]" = torch.ops.aten.permute.default(arg464_1, [1, 0]);  arg464_1 = None
    addmm_123: "f32[1568, 1280]" = torch.ops.aten.addmm.default(arg465_1, view_385, permute_262);  arg465_1 = view_385 = permute_262 = None
    view_386: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_123, [8, 196, 1280]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_228: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, 0.5)
    mul_229: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, 0.7071067811865476);  view_386 = None
    erf_24: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
    add_232: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_230: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_228, add_232);  mul_228 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_81: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_230);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_387: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_81, [1568, 1280]);  clone_81 = None
    permute_263: "f32[1280, 320]" = torch.ops.aten.permute.default(arg466_1, [1, 0]);  arg466_1 = None
    addmm_124: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg467_1, view_387, permute_263);  arg467_1 = view_387 = permute_263 = None
    view_388: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_124, [8, 196, 320]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_82: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_388);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_233: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_229, clone_82);  add_229 = clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    view_389: "f32[8, 14, 14, 320]" = torch.ops.aten.view.default(add_233, [8, 14, 14, -1]);  add_233 = None
    permute_264: "f32[8, 320, 14, 14]" = torch.ops.aten.permute.default(view_389, [0, 3, 1, 2]);  view_389 = None
    clone_83: "f32[8, 320, 14, 14]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution_31: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(clone_83, arg468_1, arg469_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_83 = arg468_1 = arg469_1 = None
    view_390: "f32[8, 512, 49]" = torch.ops.aten.view.default(convolution_31, [8, 512, 49]);  convolution_31 = None
    permute_265: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_390, [0, 2, 1]);  view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_84: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    var_mean_78 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_431: "f32[8, 49, 1]" = var_mean_78[0]
    getitem_432: "f32[8, 49, 1]" = var_mean_78[1];  var_mean_78 = None
    add_234: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_431, 1e-05);  getitem_431 = None
    rsqrt_78: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    sub_78: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_84, getitem_432);  clone_84 = getitem_432 = None
    mul_231: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
    mul_232: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_231, arg470_1);  mul_231 = arg470_1 = None
    add_235: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_232, arg471_1);  mul_232 = arg471_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_85: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_235);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_79 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_433: "f32[8, 49, 1]" = var_mean_79[0]
    getitem_434: "f32[8, 49, 1]" = var_mean_79[1];  var_mean_79 = None
    add_236: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_433, 1e-06);  getitem_433 = None
    rsqrt_79: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    sub_79: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_85, getitem_434);  getitem_434 = None
    mul_233: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
    mul_234: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_233, arg472_1);  mul_233 = arg472_1 = None
    add_237: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_234, arg473_1);  mul_234 = arg473_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_391: "f32[392, 512]" = torch.ops.aten.view.default(add_237, [392, 512])
    permute_266: "f32[512, 512]" = torch.ops.aten.permute.default(arg474_1, [1, 0]);  arg474_1 = None
    addmm_125: "f32[392, 512]" = torch.ops.aten.addmm.default(arg475_1, view_391, permute_266);  arg475_1 = view_391 = permute_266 = None
    view_392: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_125, [8, 49, 512]);  addmm_125 = None
    view_393: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_392, [8, 49, 8, 64]);  view_392 = None
    permute_267: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_394: "f32[392, 512]" = torch.ops.aten.view.default(add_237, [392, 512]);  add_237 = None
    permute_268: "f32[512, 1024]" = torch.ops.aten.permute.default(arg476_1, [1, 0]);  arg476_1 = None
    addmm_126: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg477_1, view_394, permute_268);  arg477_1 = view_394 = permute_268 = None
    view_395: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_126, [8, 49, 1024]);  addmm_126 = None
    view_396: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.view.default(view_395, [8, -1, 2, 8, 64]);  view_395 = None
    permute_269: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_396, [2, 0, 3, 1, 4]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_25 = torch.ops.aten.unbind.int(permute_269);  permute_269 = None
    getitem_435: "f32[8, 8, 49, 64]" = unbind_25[0]
    getitem_436: "f32[8, 8, 49, 64]" = unbind_25[1];  unbind_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_25 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_267, getitem_435, getitem_436);  permute_267 = getitem_435 = getitem_436 = None
    getitem_437: "f32[8, 8, 49, 64]" = _scaled_dot_product_flash_attention_25[0];  _scaled_dot_product_flash_attention_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_270: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_437, [0, 2, 1, 3]);  getitem_437 = None
    view_397: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_270, [8, 49, 512]);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_398: "f32[392, 512]" = torch.ops.aten.view.default(view_397, [392, 512]);  view_397 = None
    permute_271: "f32[512, 512]" = torch.ops.aten.permute.default(arg478_1, [1, 0]);  arg478_1 = None
    addmm_127: "f32[392, 512]" = torch.ops.aten.addmm.default(arg479_1, view_398, permute_271);  arg479_1 = view_398 = permute_271 = None
    view_399: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_127, [8, 49, 512]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_86: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_399);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_238: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(clone_85, clone_86);  clone_85 = clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_80 = torch.ops.aten.var_mean.correction(add_238, [2], correction = 0, keepdim = True)
    getitem_446: "f32[8, 49, 1]" = var_mean_80[0]
    getitem_447: "f32[8, 49, 1]" = var_mean_80[1];  var_mean_80 = None
    add_239: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_446, 1e-06);  getitem_446 = None
    rsqrt_80: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    sub_80: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_238, getitem_447);  getitem_447 = None
    mul_235: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
    mul_236: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_235, arg480_1);  mul_235 = arg480_1 = None
    add_240: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_236, arg481_1);  mul_236 = arg481_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_400: "f32[392, 512]" = torch.ops.aten.view.default(add_240, [392, 512]);  add_240 = None
    permute_272: "f32[512, 2048]" = torch.ops.aten.permute.default(arg482_1, [1, 0]);  arg482_1 = None
    addmm_128: "f32[392, 2048]" = torch.ops.aten.addmm.default(arg483_1, view_400, permute_272);  arg483_1 = view_400 = permute_272 = None
    view_401: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_128, [8, 49, 2048]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_237: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, 0.5)
    mul_238: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, 0.7071067811865476);  view_401 = None
    erf_25: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
    add_241: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_239: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_237, add_241);  mul_237 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_87: "f32[8, 49, 2048]" = torch.ops.aten.clone.default(mul_239);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_402: "f32[392, 2048]" = torch.ops.aten.view.default(clone_87, [392, 2048]);  clone_87 = None
    permute_273: "f32[2048, 512]" = torch.ops.aten.permute.default(arg484_1, [1, 0]);  arg484_1 = None
    addmm_129: "f32[392, 512]" = torch.ops.aten.addmm.default(arg485_1, view_402, permute_273);  arg485_1 = view_402 = permute_273 = None
    view_403: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_129, [8, 49, 512]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_88: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_403);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_242: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_238, clone_88);  add_238 = clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_274: "f32[8, 512, 49]" = torch.ops.aten.permute.default(add_242, [0, 2, 1]);  add_242 = None
    view_404: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_274, [8, 512, 7, 7]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_32: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_404, arg486_1, arg487_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  arg486_1 = arg487_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_243: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_32, view_404);  convolution_32 = view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_406: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_243, [8, 512, 49]);  add_243 = None
    permute_276: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
    var_mean_81 = torch.ops.aten.var_mean.correction(permute_276, [2], correction = 0, keepdim = True)
    getitem_448: "f32[8, 49, 1]" = var_mean_81[0]
    getitem_449: "f32[8, 49, 1]" = var_mean_81[1];  var_mean_81 = None
    add_244: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_448, 1e-06);  getitem_448 = None
    rsqrt_81: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_81: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(permute_276, getitem_449);  getitem_449 = None
    mul_240: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
    mul_241: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_240, arg488_1);  mul_240 = arg488_1 = None
    add_245: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_241, arg489_1);  mul_241 = arg489_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_407: "f32[392, 512]" = torch.ops.aten.view.default(add_245, [392, 512])
    permute_277: "f32[512, 512]" = torch.ops.aten.permute.default(arg490_1, [1, 0]);  arg490_1 = None
    addmm_130: "f32[392, 512]" = torch.ops.aten.addmm.default(arg491_1, view_407, permute_277);  arg491_1 = view_407 = permute_277 = None
    view_408: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_130, [8, 49, 512]);  addmm_130 = None
    view_409: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_408, [8, 49, 8, 64]);  view_408 = None
    permute_278: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_410: "f32[392, 512]" = torch.ops.aten.view.default(add_245, [392, 512]);  add_245 = None
    permute_279: "f32[512, 1024]" = torch.ops.aten.permute.default(arg492_1, [1, 0]);  arg492_1 = None
    addmm_131: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg493_1, view_410, permute_279);  arg493_1 = view_410 = permute_279 = None
    view_411: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_131, [8, 49, 1024]);  addmm_131 = None
    view_412: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.view.default(view_411, [8, -1, 2, 8, 64]);  view_411 = None
    permute_280: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_412, [2, 0, 3, 1, 4]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_26 = torch.ops.aten.unbind.int(permute_280);  permute_280 = None
    getitem_450: "f32[8, 8, 49, 64]" = unbind_26[0]
    getitem_451: "f32[8, 8, 49, 64]" = unbind_26[1];  unbind_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_26 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_278, getitem_450, getitem_451);  permute_278 = getitem_450 = getitem_451 = None
    getitem_452: "f32[8, 8, 49, 64]" = _scaled_dot_product_flash_attention_26[0];  _scaled_dot_product_flash_attention_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_281: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_452, [0, 2, 1, 3]);  getitem_452 = None
    view_413: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_281, [8, 49, 512]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_414: "f32[392, 512]" = torch.ops.aten.view.default(view_413, [392, 512]);  view_413 = None
    permute_282: "f32[512, 512]" = torch.ops.aten.permute.default(arg494_1, [1, 0]);  arg494_1 = None
    addmm_132: "f32[392, 512]" = torch.ops.aten.addmm.default(arg495_1, view_414, permute_282);  arg495_1 = view_414 = permute_282 = None
    view_415: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_132, [8, 49, 512]);  addmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_89: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_415);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_246: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(permute_276, clone_89);  permute_276 = clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_82 = torch.ops.aten.var_mean.correction(add_246, [2], correction = 0, keepdim = True)
    getitem_461: "f32[8, 49, 1]" = var_mean_82[0]
    getitem_462: "f32[8, 49, 1]" = var_mean_82[1];  var_mean_82 = None
    add_247: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_461, 1e-06);  getitem_461 = None
    rsqrt_82: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_82: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_246, getitem_462);  getitem_462 = None
    mul_242: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
    mul_243: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_242, arg496_1);  mul_242 = arg496_1 = None
    add_248: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_243, arg497_1);  mul_243 = arg497_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_416: "f32[392, 512]" = torch.ops.aten.view.default(add_248, [392, 512]);  add_248 = None
    permute_283: "f32[512, 2048]" = torch.ops.aten.permute.default(arg498_1, [1, 0]);  arg498_1 = None
    addmm_133: "f32[392, 2048]" = torch.ops.aten.addmm.default(arg499_1, view_416, permute_283);  arg499_1 = view_416 = permute_283 = None
    view_417: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_133, [8, 49, 2048]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_244: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
    mul_245: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, 0.7071067811865476);  view_417 = None
    erf_26: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_245);  mul_245 = None
    add_249: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_246: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_244, add_249);  mul_244 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_90: "f32[8, 49, 2048]" = torch.ops.aten.clone.default(mul_246);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_418: "f32[392, 2048]" = torch.ops.aten.view.default(clone_90, [392, 2048]);  clone_90 = None
    permute_284: "f32[2048, 512]" = torch.ops.aten.permute.default(arg500_1, [1, 0]);  arg500_1 = None
    addmm_134: "f32[392, 512]" = torch.ops.aten.addmm.default(arg501_1, view_418, permute_284);  arg501_1 = view_418 = permute_284 = None
    view_419: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_134, [8, 49, 512]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_91: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_419);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_250: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_246, clone_91);  add_246 = clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_83 = torch.ops.aten.var_mean.correction(add_250, [2], correction = 0, keepdim = True)
    getitem_463: "f32[8, 49, 1]" = var_mean_83[0]
    getitem_464: "f32[8, 49, 1]" = var_mean_83[1];  var_mean_83 = None
    add_251: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_463, 1e-06);  getitem_463 = None
    rsqrt_83: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_83: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_250, getitem_464);  getitem_464 = None
    mul_247: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
    mul_248: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_247, arg502_1);  mul_247 = arg502_1 = None
    add_252: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_248, arg503_1);  mul_248 = arg503_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_420: "f32[392, 512]" = torch.ops.aten.view.default(add_252, [392, 512])
    permute_285: "f32[512, 512]" = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
    addmm_135: "f32[392, 512]" = torch.ops.aten.addmm.default(arg505_1, view_420, permute_285);  arg505_1 = view_420 = permute_285 = None
    view_421: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_135, [8, 49, 512]);  addmm_135 = None
    view_422: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_421, [8, 49, 8, 64]);  view_421 = None
    permute_286: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_423: "f32[392, 512]" = torch.ops.aten.view.default(add_252, [392, 512]);  add_252 = None
    permute_287: "f32[512, 1024]" = torch.ops.aten.permute.default(arg506_1, [1, 0]);  arg506_1 = None
    addmm_136: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg507_1, view_423, permute_287);  arg507_1 = view_423 = permute_287 = None
    view_424: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_136, [8, 49, 1024]);  addmm_136 = None
    view_425: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.view.default(view_424, [8, -1, 2, 8, 64]);  view_424 = None
    permute_288: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_425, [2, 0, 3, 1, 4]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_27 = torch.ops.aten.unbind.int(permute_288);  permute_288 = None
    getitem_465: "f32[8, 8, 49, 64]" = unbind_27[0]
    getitem_466: "f32[8, 8, 49, 64]" = unbind_27[1];  unbind_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_27 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_286, getitem_465, getitem_466);  permute_286 = getitem_465 = getitem_466 = None
    getitem_467: "f32[8, 8, 49, 64]" = _scaled_dot_product_flash_attention_27[0];  _scaled_dot_product_flash_attention_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_289: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_467, [0, 2, 1, 3]);  getitem_467 = None
    view_426: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_289, [8, 49, 512]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_427: "f32[392, 512]" = torch.ops.aten.view.default(view_426, [392, 512]);  view_426 = None
    permute_290: "f32[512, 512]" = torch.ops.aten.permute.default(arg508_1, [1, 0]);  arg508_1 = None
    addmm_137: "f32[392, 512]" = torch.ops.aten.addmm.default(arg509_1, view_427, permute_290);  arg509_1 = view_427 = permute_290 = None
    view_428: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_137, [8, 49, 512]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_92: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_428);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_253: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_250, clone_92);  add_250 = clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_84 = torch.ops.aten.var_mean.correction(add_253, [2], correction = 0, keepdim = True)
    getitem_476: "f32[8, 49, 1]" = var_mean_84[0]
    getitem_477: "f32[8, 49, 1]" = var_mean_84[1];  var_mean_84 = None
    add_254: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_476, 1e-06);  getitem_476 = None
    rsqrt_84: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
    sub_84: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_253, getitem_477);  getitem_477 = None
    mul_249: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
    mul_250: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_249, arg510_1);  mul_249 = arg510_1 = None
    add_255: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_250, arg511_1);  mul_250 = arg511_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_429: "f32[392, 512]" = torch.ops.aten.view.default(add_255, [392, 512]);  add_255 = None
    permute_291: "f32[512, 2048]" = torch.ops.aten.permute.default(arg512_1, [1, 0]);  arg512_1 = None
    addmm_138: "f32[392, 2048]" = torch.ops.aten.addmm.default(arg513_1, view_429, permute_291);  arg513_1 = view_429 = permute_291 = None
    view_430: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_138, [8, 49, 2048]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_251: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, 0.5)
    mul_252: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, 0.7071067811865476);  view_430 = None
    erf_27: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
    add_256: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_253: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_251, add_256);  mul_251 = add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_93: "f32[8, 49, 2048]" = torch.ops.aten.clone.default(mul_253);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_431: "f32[392, 2048]" = torch.ops.aten.view.default(clone_93, [392, 2048]);  clone_93 = None
    permute_292: "f32[2048, 512]" = torch.ops.aten.permute.default(arg514_1, [1, 0]);  arg514_1 = None
    addmm_139: "f32[392, 512]" = torch.ops.aten.addmm.default(arg515_1, view_431, permute_292);  arg515_1 = view_431 = permute_292 = None
    view_432: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_139, [8, 49, 512]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_94: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_432);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_257: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_253, clone_94);  add_253 = clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:416, code: x = self.norm(x)
    var_mean_85 = torch.ops.aten.var_mean.correction(add_257, [2], correction = 0, keepdim = True)
    getitem_478: "f32[8, 49, 1]" = var_mean_85[0]
    getitem_479: "f32[8, 49, 1]" = var_mean_85[1];  var_mean_85 = None
    add_258: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_478, 1e-06);  getitem_478 = None
    rsqrt_85: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_85: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_257, getitem_479);  add_257 = getitem_479 = None
    mul_254: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
    mul_255: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_254, arg516_1);  mul_254 = arg516_1 = None
    add_259: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_255, arg517_1);  mul_255 = arg517_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:421, code: x = x.mean(dim=1)
    mean: "f32[8, 512]" = torch.ops.aten.mean.dim(add_259, [1]);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:422, code: x = self.head_drop(x)
    clone_95: "f32[8, 512]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:423, code: return x if pre_logits else self.head(x)
    permute_293: "f32[512, 1000]" = torch.ops.aten.permute.default(arg518_1, [1, 0]);  arg518_1 = None
    addmm_140: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg519_1, clone_95, permute_293);  arg519_1 = clone_95 = permute_293 = None
    return (addmm_140,)
    