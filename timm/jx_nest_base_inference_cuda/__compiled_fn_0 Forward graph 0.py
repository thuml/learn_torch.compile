from __future__ import annotations



def forward(self, arg0_1: "f32[1, 16, 196, 128]", arg1_1: "f32[128]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[256]", arg10_1: "f32[256]", arg11_1: "f32[1, 4, 196, 256]", arg12_1: "f32[256]", arg13_1: "f32[256]", arg14_1: "f32[256]", arg15_1: "f32[256]", arg16_1: "f32[256]", arg17_1: "f32[256]", arg18_1: "f32[256]", arg19_1: "f32[256]", arg20_1: "f32[512]", arg21_1: "f32[512]", arg22_1: "f32[1, 1, 196, 512]", arg23_1: "f32[512]", arg24_1: "f32[512]", arg25_1: "f32[512]", arg26_1: "f32[512]", arg27_1: "f32[512]", arg28_1: "f32[512]", arg29_1: "f32[512]", arg30_1: "f32[512]", arg31_1: "f32[512]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[512]", arg37_1: "f32[512]", arg38_1: "f32[512]", arg39_1: "f32[512]", arg40_1: "f32[512]", arg41_1: "f32[512]", arg42_1: "f32[512]", arg43_1: "f32[512]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[512]", arg47_1: "f32[512]", arg48_1: "f32[512]", arg49_1: "f32[512]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[512]", arg54_1: "f32[512]", arg55_1: "f32[512]", arg56_1: "f32[512]", arg57_1: "f32[512]", arg58_1: "f32[512]", arg59_1: "f32[512]", arg60_1: "f32[512]", arg61_1: "f32[512]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[512]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[512]", arg72_1: "f32[512]", arg73_1: "f32[512]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[512]", arg79_1: "f32[512]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512]", arg86_1: "f32[512]", arg87_1: "f32[512]", arg88_1: "f32[512]", arg89_1: "f32[512]", arg90_1: "f32[512]", arg91_1: "f32[512]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[512]", arg95_1: "f32[512]", arg96_1: "f32[512]", arg97_1: "f32[512]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[512]", arg101_1: "f32[512]", arg102_1: "f32[512]", arg103_1: "f32[512]", arg104_1: "f32[512]", arg105_1: "f32[128, 3, 4, 4]", arg106_1: "f32[128]", arg107_1: "f32[384, 128]", arg108_1: "f32[384]", arg109_1: "f32[128, 128]", arg110_1: "f32[128]", arg111_1: "f32[512, 128]", arg112_1: "f32[512]", arg113_1: "f32[128, 512]", arg114_1: "f32[128]", arg115_1: "f32[384, 128]", arg116_1: "f32[384]", arg117_1: "f32[128, 128]", arg118_1: "f32[128]", arg119_1: "f32[512, 128]", arg120_1: "f32[512]", arg121_1: "f32[128, 512]", arg122_1: "f32[128]", arg123_1: "f32[256, 128, 3, 3]", arg124_1: "f32[256]", arg125_1: "f32[768, 256]", arg126_1: "f32[768]", arg127_1: "f32[256, 256]", arg128_1: "f32[256]", arg129_1: "f32[1024, 256]", arg130_1: "f32[1024]", arg131_1: "f32[256, 1024]", arg132_1: "f32[256]", arg133_1: "f32[768, 256]", arg134_1: "f32[768]", arg135_1: "f32[256, 256]", arg136_1: "f32[256]", arg137_1: "f32[1024, 256]", arg138_1: "f32[1024]", arg139_1: "f32[256, 1024]", arg140_1: "f32[256]", arg141_1: "f32[512, 256, 3, 3]", arg142_1: "f32[512]", arg143_1: "f32[1536, 512]", arg144_1: "f32[1536]", arg145_1: "f32[512, 512]", arg146_1: "f32[512]", arg147_1: "f32[2048, 512]", arg148_1: "f32[2048]", arg149_1: "f32[512, 2048]", arg150_1: "f32[512]", arg151_1: "f32[1536, 512]", arg152_1: "f32[1536]", arg153_1: "f32[512, 512]", arg154_1: "f32[512]", arg155_1: "f32[2048, 512]", arg156_1: "f32[2048]", arg157_1: "f32[512, 2048]", arg158_1: "f32[512]", arg159_1: "f32[1536, 512]", arg160_1: "f32[1536]", arg161_1: "f32[512, 512]", arg162_1: "f32[512]", arg163_1: "f32[2048, 512]", arg164_1: "f32[2048]", arg165_1: "f32[512, 2048]", arg166_1: "f32[512]", arg167_1: "f32[1536, 512]", arg168_1: "f32[1536]", arg169_1: "f32[512, 512]", arg170_1: "f32[512]", arg171_1: "f32[2048, 512]", arg172_1: "f32[2048]", arg173_1: "f32[512, 2048]", arg174_1: "f32[512]", arg175_1: "f32[1536, 512]", arg176_1: "f32[1536]", arg177_1: "f32[512, 512]", arg178_1: "f32[512]", arg179_1: "f32[2048, 512]", arg180_1: "f32[2048]", arg181_1: "f32[512, 2048]", arg182_1: "f32[512]", arg183_1: "f32[1536, 512]", arg184_1: "f32[1536]", arg185_1: "f32[512, 512]", arg186_1: "f32[512]", arg187_1: "f32[2048, 512]", arg188_1: "f32[2048]", arg189_1: "f32[512, 2048]", arg190_1: "f32[512]", arg191_1: "f32[1536, 512]", arg192_1: "f32[1536]", arg193_1: "f32[512, 512]", arg194_1: "f32[512]", arg195_1: "f32[2048, 512]", arg196_1: "f32[2048]", arg197_1: "f32[512, 2048]", arg198_1: "f32[512]", arg199_1: "f32[1536, 512]", arg200_1: "f32[1536]", arg201_1: "f32[512, 512]", arg202_1: "f32[512]", arg203_1: "f32[2048, 512]", arg204_1: "f32[2048]", arg205_1: "f32[512, 2048]", arg206_1: "f32[512]", arg207_1: "f32[1536, 512]", arg208_1: "f32[1536]", arg209_1: "f32[512, 512]", arg210_1: "f32[512]", arg211_1: "f32[2048, 512]", arg212_1: "f32[2048]", arg213_1: "f32[512, 2048]", arg214_1: "f32[512]", arg215_1: "f32[1536, 512]", arg216_1: "f32[1536]", arg217_1: "f32[512, 512]", arg218_1: "f32[512]", arg219_1: "f32[2048, 512]", arg220_1: "f32[2048]", arg221_1: "f32[512, 2048]", arg222_1: "f32[512]", arg223_1: "f32[1536, 512]", arg224_1: "f32[1536]", arg225_1: "f32[512, 512]", arg226_1: "f32[512]", arg227_1: "f32[2048, 512]", arg228_1: "f32[2048]", arg229_1: "f32[512, 2048]", arg230_1: "f32[512]", arg231_1: "f32[1536, 512]", arg232_1: "f32[1536]", arg233_1: "f32[512, 512]", arg234_1: "f32[512]", arg235_1: "f32[2048, 512]", arg236_1: "f32[2048]", arg237_1: "f32[512, 2048]", arg238_1: "f32[512]", arg239_1: "f32[1536, 512]", arg240_1: "f32[1536]", arg241_1: "f32[512, 512]", arg242_1: "f32[512]", arg243_1: "f32[2048, 512]", arg244_1: "f32[2048]", arg245_1: "f32[512, 2048]", arg246_1: "f32[512]", arg247_1: "f32[1536, 512]", arg248_1: "f32[1536]", arg249_1: "f32[512, 512]", arg250_1: "f32[512]", arg251_1: "f32[2048, 512]", arg252_1: "f32[2048]", arg253_1: "f32[512, 2048]", arg254_1: "f32[512]", arg255_1: "f32[1536, 512]", arg256_1: "f32[1536]", arg257_1: "f32[512, 512]", arg258_1: "f32[512]", arg259_1: "f32[2048, 512]", arg260_1: "f32[2048]", arg261_1: "f32[512, 2048]", arg262_1: "f32[512]", arg263_1: "f32[1536, 512]", arg264_1: "f32[1536]", arg265_1: "f32[512, 512]", arg266_1: "f32[512]", arg267_1: "f32[2048, 512]", arg268_1: "f32[2048]", arg269_1: "f32[512, 2048]", arg270_1: "f32[512]", arg271_1: "f32[1536, 512]", arg272_1: "f32[1536]", arg273_1: "f32[512, 512]", arg274_1: "f32[512]", arg275_1: "f32[2048, 512]", arg276_1: "f32[2048]", arg277_1: "f32[512, 2048]", arg278_1: "f32[512]", arg279_1: "f32[1536, 512]", arg280_1: "f32[1536]", arg281_1: "f32[512, 512]", arg282_1: "f32[512]", arg283_1: "f32[2048, 512]", arg284_1: "f32[2048]", arg285_1: "f32[512, 2048]", arg286_1: "f32[512]", arg287_1: "f32[1536, 512]", arg288_1: "f32[1536]", arg289_1: "f32[512, 512]", arg290_1: "f32[512]", arg291_1: "f32[2048, 512]", arg292_1: "f32[2048]", arg293_1: "f32[512, 2048]", arg294_1: "f32[512]", arg295_1: "f32[1536, 512]", arg296_1: "f32[1536]", arg297_1: "f32[512, 512]", arg298_1: "f32[512]", arg299_1: "f32[2048, 512]", arg300_1: "f32[2048]", arg301_1: "f32[512, 2048]", arg302_1: "f32[512]", arg303_1: "f32[1000, 512]", arg304_1: "f32[1000]", arg305_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(arg305_1, arg105_1, arg106_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg305_1 = arg105_1 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.view.default(permute, [8, 4, 14, 4, 14, 128]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_1: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
    clone: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone, [8, 16, 196, 128]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(view_1, arg0_1);  view_1 = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean = torch.ops.aten.var_mean.correction(add, [3], correction = 0, keepdim = True)
    getitem: "f32[8, 16, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 16, 196, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    mul: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
    add_2: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_2: "f32[25088, 128]" = torch.ops.aten.view.default(add_2, [25088, 128]);  add_2 = None
    permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm: "f32[25088, 384]" = torch.ops.aten.addmm.default(arg108_1, view_2, permute_2);  arg108_1 = view_2 = permute_2 = None
    view_3: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(addmm, [8, 16, 196, 384]);  addmm = None
    view_4: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.view.default(view_3, [8, 16, 196, 3, 4, 32]);  view_3 = None
    permute_3: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_4, [3, 0, 4, 1, 2, 5]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_2: "f32[8, 4, 16, 196, 32]" = unbind[0]
    getitem_3: "f32[8, 4, 16, 196, 32]" = unbind[1]
    getitem_4: "f32[8, 4, 16, 196, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_2: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_2, 0.42044820762685725);  getitem_2 = None
    permute_4: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 2, 4, 3]);  getitem_3 = None
    mul_3: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_4, 0.42044820762685725);  permute_4 = None
    expand: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_2, [8, 4, 16, 196, 32]);  mul_2 = None
    clone_1: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_5: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_1, [512, 196, 32]);  clone_1 = None
    expand_1: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_3, [8, 4, 16, 32, 196]);  mul_3 = None
    clone_2: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_6: "f32[512, 32, 196]" = torch.ops.aten.view.default(clone_2, [512, 32, 196]);  clone_2 = None
    bmm: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
    view_7: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 4, 16, 196, 196]);  bmm = None
    amax: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_7, [-1], True)
    sub_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_7, amax);  view_7 = amax = None
    exp: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    expand_2: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(div, [8, 4, 16, 196, 196]);  div = None
    view_8: "f32[512, 196, 196]" = torch.ops.aten.view.default(expand_2, [512, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_4, [8, 4, 16, 196, 32]);  getitem_4 = None
    clone_3: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_9: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_3, [512, 196, 32]);  clone_3 = None
    bmm_1: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = view_9 = None
    view_10: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_1, [8, 4, 16, 196, 32]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_5: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_10, [0, 2, 3, 4, 1]);  view_10 = None
    clone_4: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_11: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone_4, [8, 16, 196, 128]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_12: "f32[25088, 128]" = torch.ops.aten.view.default(view_11, [25088, 128]);  view_11 = None
    permute_6: "f32[128, 128]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_1: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg110_1, view_12, permute_6);  arg110_1 = view_12 = permute_6 = None
    view_13: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_1, [8, 16, 196, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_5: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_13);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_3: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add, clone_5);  add = clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [3], correction = 0, keepdim = True)
    getitem_5: "f32[8, 16, 196, 1]" = var_mean_1[0]
    getitem_6: "f32[8, 16, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
    rsqrt_1: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
    mul_4: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_4, arg3_1);  mul_4 = arg3_1 = None
    add_5: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_5, arg4_1);  mul_5 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[25088, 128]" = torch.ops.aten.view.default(add_5, [25088, 128]);  add_5 = None
    permute_7: "f32[128, 512]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    addmm_2: "f32[25088, 512]" = torch.ops.aten.addmm.default(arg112_1, view_14, permute_7);  arg112_1 = view_14 = permute_7 = None
    view_15: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(addmm_2, [8, 16, 196, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_6: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_7: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
    erf: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_6: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 16, 196, 512]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[25088, 512]" = torch.ops.aten.view.default(clone_6, [25088, 512]);  clone_6 = None
    permute_8: "f32[512, 128]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    addmm_3: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg114_1, view_16, permute_8);  arg114_1 = view_16 = permute_8 = None
    view_17: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_3, [8, 16, 196, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_7: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_3, clone_7);  add_3 = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [3], correction = 0, keepdim = True)
    getitem_7: "f32[8, 16, 196, 1]" = var_mean_2[0]
    getitem_8: "f32[8, 16, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
    rsqrt_2: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_7, getitem_8);  getitem_8 = None
    mul_9: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_9, arg5_1);  mul_9 = arg5_1 = None
    add_9: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_10, arg6_1);  mul_10 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_18: "f32[25088, 128]" = torch.ops.aten.view.default(add_9, [25088, 128]);  add_9 = None
    permute_9: "f32[128, 384]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_4: "f32[25088, 384]" = torch.ops.aten.addmm.default(arg116_1, view_18, permute_9);  arg116_1 = view_18 = permute_9 = None
    view_19: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(addmm_4, [8, 16, 196, 384]);  addmm_4 = None
    view_20: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.view.default(view_19, [8, 16, 196, 3, 4, 32]);  view_19 = None
    permute_10: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_20, [3, 0, 4, 1, 2, 5]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_10);  permute_10 = None
    getitem_9: "f32[8, 4, 16, 196, 32]" = unbind_1[0]
    getitem_10: "f32[8, 4, 16, 196, 32]" = unbind_1[1]
    getitem_11: "f32[8, 4, 16, 196, 32]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_11: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_9, 0.42044820762685725);  getitem_9 = None
    permute_11: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 2, 4, 3]);  getitem_10 = None
    mul_12: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_11, 0.42044820762685725);  permute_11 = None
    expand_4: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_11, [8, 4, 16, 196, 32]);  mul_11 = None
    clone_8: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_21: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_8, [512, 196, 32]);  clone_8 = None
    expand_5: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_12, [8, 4, 16, 32, 196]);  mul_12 = None
    clone_9: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_22: "f32[512, 32, 196]" = torch.ops.aten.view.default(clone_9, [512, 32, 196]);  clone_9 = None
    bmm_2: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_21, view_22);  view_21 = view_22 = None
    view_23: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 4, 16, 196, 196]);  bmm_2 = None
    amax_1: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_23, [-1], True)
    sub_4: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_23, amax_1);  view_23 = amax_1 = None
    exp_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    expand_6: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(div_1, [8, 4, 16, 196, 196]);  div_1 = None
    view_24: "f32[512, 196, 196]" = torch.ops.aten.view.default(expand_6, [512, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_11, [8, 4, 16, 196, 32]);  getitem_11 = None
    clone_10: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_25: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_10, [512, 196, 32]);  clone_10 = None
    bmm_3: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = view_25 = None
    view_26: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_3, [8, 4, 16, 196, 32]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_12: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_26, [0, 2, 3, 4, 1]);  view_26 = None
    clone_11: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_27: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone_11, [8, 16, 196, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_28: "f32[25088, 128]" = torch.ops.aten.view.default(view_27, [25088, 128]);  view_27 = None
    permute_13: "f32[128, 128]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_5: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg118_1, view_28, permute_13);  arg118_1 = view_28 = permute_13 = None
    view_29: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_5, [8, 16, 196, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_12: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_29);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_10: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_7, clone_12);  add_7 = clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 16, 196, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 16, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_10, getitem_13);  getitem_13 = None
    mul_13: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_14: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_13, arg7_1);  mul_13 = arg7_1 = None
    add_12: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_14, arg8_1);  mul_14 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[25088, 128]" = torch.ops.aten.view.default(add_12, [25088, 128]);  add_12 = None
    permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_6: "f32[25088, 512]" = torch.ops.aten.addmm.default(arg120_1, view_30, permute_14);  arg120_1 = view_30 = permute_14 = None
    view_31: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(addmm_6, [8, 16, 196, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_15: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_16: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476);  view_31 = None
    erf_1: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
    add_13: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_17: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_15, add_13);  mul_15 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 16, 196, 512]" = torch.ops.aten.clone.default(mul_17);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[25088, 512]" = torch.ops.aten.view.default(clone_13, [25088, 512]);  clone_13 = None
    permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_7: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg122_1, view_32, permute_15);  arg122_1 = view_32 = permute_15 = None
    view_33: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_7, [8, 16, 196, 128]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_14: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_10, clone_14);  add_10 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_34: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.view.default(add_14, [8, 4, 4, 14, 14, 128]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_16: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.permute.default(view_34, [0, 1, 3, 2, 4, 5]);  view_34 = None
    clone_15: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_35: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_15, [8, 56, 56, 128]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_17: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_35, [0, 3, 1, 2]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_1: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(permute_17, arg123_1, arg124_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_17 = arg123_1 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_18: "f32[8, 56, 56, 256]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_16: "f32[8, 56, 56, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_16, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 56, 56, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 56, 56, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(clone_16, getitem_15);  clone_16 = getitem_15 = None
    mul_18: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_19: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_18, arg9_1);  mul_18 = arg9_1 = None
    add_16: "f32[8, 56, 56, 256]" = torch.ops.aten.add.Tensor(mul_19, arg10_1);  mul_19 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_19: "f32[8, 256, 56, 56]" = torch.ops.aten.permute.default(add_16, [0, 3, 1, 2]);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 256, 57, 57]" = torch.ops.aten.constant_pad_nd.default(permute_19, [0, 1, 0, 1], -inf);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd, [3, 3], [2, 2]);  constant_pad_nd = None
    getitem_16: "f32[8, 256, 28, 28]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_20: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 3, 1]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_36: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.view.default(permute_20, [8, 2, 14, 2, 14, 256]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_21: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.permute.default(view_36, [0, 1, 3, 2, 4, 5]);  view_36 = None
    clone_17: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_37: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_17, [8, 4, 196, 256]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add_17: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(view_37, arg11_1);  view_37 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 4, 196, 1]" = var_mean_5[0]
    getitem_19: "f32[8, 4, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_5: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_19);  getitem_19 = None
    mul_20: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
    mul_21: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_20, arg12_1);  mul_20 = arg12_1 = None
    add_19: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_21, arg13_1);  mul_21 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_38: "f32[6272, 256]" = torch.ops.aten.view.default(add_19, [6272, 256]);  add_19 = None
    permute_22: "f32[256, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_8: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg126_1, view_38, permute_22);  arg126_1 = view_38 = permute_22 = None
    view_39: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(addmm_8, [8, 4, 196, 768]);  addmm_8 = None
    view_40: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.view.default(view_39, [8, 4, 196, 3, 8, 32]);  view_39 = None
    permute_23: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_40, [3, 0, 4, 1, 2, 5]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
    getitem_20: "f32[8, 8, 4, 196, 32]" = unbind_2[0]
    getitem_21: "f32[8, 8, 4, 196, 32]" = unbind_2[1]
    getitem_22: "f32[8, 8, 4, 196, 32]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_22: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_20, 0.42044820762685725);  getitem_20 = None
    permute_24: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 2, 4, 3]);  getitem_21 = None
    mul_23: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_24, 0.42044820762685725);  permute_24 = None
    expand_8: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_22, [8, 8, 4, 196, 32]);  mul_22 = None
    clone_18: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_41: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_18, [256, 196, 32]);  clone_18 = None
    expand_9: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_23, [8, 8, 4, 32, 196]);  mul_23 = None
    clone_19: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_42: "f32[256, 32, 196]" = torch.ops.aten.view.default(clone_19, [256, 32, 196]);  clone_19 = None
    bmm_4: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_41, view_42);  view_41 = view_42 = None
    view_43: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 8, 4, 196, 196]);  bmm_4 = None
    amax_2: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_43, [-1], True)
    sub_8: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_43, amax_2);  view_43 = amax_2 = None
    exp_2: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    expand_10: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(div_2, [8, 8, 4, 196, 196]);  div_2 = None
    view_44: "f32[256, 196, 196]" = torch.ops.aten.view.default(expand_10, [256, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_22, [8, 8, 4, 196, 32]);  getitem_22 = None
    clone_20: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_45: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_20, [256, 196, 32]);  clone_20 = None
    bmm_5: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_44, view_45);  view_44 = view_45 = None
    view_46: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_5, [8, 8, 4, 196, 32]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_25: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_46, [0, 2, 3, 4, 1]);  view_46 = None
    clone_21: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_47: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_21, [8, 4, 196, 256]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_48: "f32[6272, 256]" = torch.ops.aten.view.default(view_47, [6272, 256]);  view_47 = None
    permute_26: "f32[256, 256]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_9: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg128_1, view_48, permute_26);  arg128_1 = view_48 = permute_26 = None
    view_49: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_9, [8, 4, 196, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_22: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_49);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_20: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_17, clone_22);  add_17 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [3], correction = 0, keepdim = True)
    getitem_23: "f32[8, 4, 196, 1]" = var_mean_6[0]
    getitem_24: "f32[8, 4, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-06);  getitem_23 = None
    rsqrt_6: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_9: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_20, getitem_24);  getitem_24 = None
    mul_24: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_25: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_24, arg14_1);  mul_24 = arg14_1 = None
    add_22: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_25, arg15_1);  mul_25 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[6272, 256]" = torch.ops.aten.view.default(add_22, [6272, 256]);  add_22 = None
    permute_27: "f32[256, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_10: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg130_1, view_50, permute_27);  arg130_1 = view_50 = permute_27 = None
    view_51: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 4, 196, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_27: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476);  view_51 = None
    erf_2: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_23: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_28: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_26, add_23);  mul_26 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 4, 196, 1024]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_23, [6272, 1024]);  clone_23 = None
    permute_28: "f32[1024, 256]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_11: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg132_1, view_52, permute_28);  arg132_1 = view_52 = permute_28 = None
    view_53: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_11, [8, 4, 196, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_53);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_24: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_20, clone_24);  add_20 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [3], correction = 0, keepdim = True)
    getitem_25: "f32[8, 4, 196, 1]" = var_mean_7[0]
    getitem_26: "f32[8, 4, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-06);  getitem_25 = None
    rsqrt_7: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_10: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_24, getitem_26);  getitem_26 = None
    mul_29: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
    mul_30: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_29, arg16_1);  mul_29 = arg16_1 = None
    add_26: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_30, arg17_1);  mul_30 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_54: "f32[6272, 256]" = torch.ops.aten.view.default(add_26, [6272, 256]);  add_26 = None
    permute_29: "f32[256, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_12: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg134_1, view_54, permute_29);  arg134_1 = view_54 = permute_29 = None
    view_55: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(addmm_12, [8, 4, 196, 768]);  addmm_12 = None
    view_56: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.view.default(view_55, [8, 4, 196, 3, 8, 32]);  view_55 = None
    permute_30: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_56, [3, 0, 4, 1, 2, 5]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
    getitem_27: "f32[8, 8, 4, 196, 32]" = unbind_3[0]
    getitem_28: "f32[8, 8, 4, 196, 32]" = unbind_3[1]
    getitem_29: "f32[8, 8, 4, 196, 32]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_31: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_27, 0.42044820762685725);  getitem_27 = None
    permute_31: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 2, 4, 3]);  getitem_28 = None
    mul_32: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_31, 0.42044820762685725);  permute_31 = None
    expand_12: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_31, [8, 8, 4, 196, 32]);  mul_31 = None
    clone_25: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_57: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_25, [256, 196, 32]);  clone_25 = None
    expand_13: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_32, [8, 8, 4, 32, 196]);  mul_32 = None
    clone_26: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_58: "f32[256, 32, 196]" = torch.ops.aten.view.default(clone_26, [256, 32, 196]);  clone_26 = None
    bmm_6: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_57, view_58);  view_57 = view_58 = None
    view_59: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 8, 4, 196, 196]);  bmm_6 = None
    amax_3: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
    sub_11: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_59, amax_3);  view_59 = amax_3 = None
    exp_3: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    expand_14: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(div_3, [8, 8, 4, 196, 196]);  div_3 = None
    view_60: "f32[256, 196, 196]" = torch.ops.aten.view.default(expand_14, [256, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_29, [8, 8, 4, 196, 32]);  getitem_29 = None
    clone_27: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_61: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_27, [256, 196, 32]);  clone_27 = None
    bmm_7: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_60, view_61);  view_60 = view_61 = None
    view_62: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_7, [8, 8, 4, 196, 32]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_32: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_62, [0, 2, 3, 4, 1]);  view_62 = None
    clone_28: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_63: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_28, [8, 4, 196, 256]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_64: "f32[6272, 256]" = torch.ops.aten.view.default(view_63, [6272, 256]);  view_63 = None
    permute_33: "f32[256, 256]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_13: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg136_1, view_64, permute_33);  arg136_1 = view_64 = permute_33 = None
    view_65: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_13, [8, 4, 196, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_29: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_27: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_24, clone_29);  add_24 = clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_27, [3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 4, 196, 1]" = var_mean_8[0]
    getitem_31: "f32[8, 4, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_8: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_12: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_27, getitem_31);  getitem_31 = None
    mul_33: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_33, arg18_1);  mul_33 = arg18_1 = None
    add_29: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_34, arg19_1);  mul_34 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_66: "f32[6272, 256]" = torch.ops.aten.view.default(add_29, [6272, 256]);  add_29 = None
    permute_34: "f32[256, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_14: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_66, permute_34);  arg138_1 = view_66 = permute_34 = None
    view_67: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 4, 196, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_35: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.5)
    mul_36: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.7071067811865476);  view_67 = None
    erf_3: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_37: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_35, add_30);  mul_35 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.clone.default(mul_37);  mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_68: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_30, [6272, 1024]);  clone_30 = None
    permute_35: "f32[1024, 256]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_15: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg140_1, view_68, permute_35);  arg140_1 = view_68 = permute_35 = None
    view_69: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_15, [8, 4, 196, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_31: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_69);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_31: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_27, clone_31);  add_27 = clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_70: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.view.default(add_31, [8, 2, 2, 14, 14, 256]);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_36: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.permute.default(view_70, [0, 1, 3, 2, 4, 5]);  view_70 = None
    clone_32: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    view_71: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_32, [8, 28, 28, 256]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_37: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_71, [0, 3, 1, 2]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_2: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_37, arg141_1, arg142_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_37 = arg141_1 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_38: "f32[8, 28, 28, 512]" = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_33: "f32[8, 28, 28, 512]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_33, [3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 28, 28, 1]" = var_mean_9[0]
    getitem_33: "f32[8, 28, 28, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_9: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_13: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(clone_33, getitem_33);  clone_33 = getitem_33 = None
    mul_38: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = rsqrt_9 = None
    mul_39: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_38, arg20_1);  mul_38 = arg20_1 = None
    add_33: "f32[8, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_39, arg21_1);  mul_39 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_39: "f32[8, 512, 28, 28]" = torch.ops.aten.permute.default(add_33, [0, 3, 1, 2]);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_1: "f32[8, 512, 29, 29]" = torch.ops.aten.constant_pad_nd.default(permute_39, [0, 1, 0, 1], -inf);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_1, [3, 3], [2, 2]);  constant_pad_nd_1 = None
    getitem_34: "f32[8, 512, 14, 14]" = max_pool2d_with_indices_1[0];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_40: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 3, 1]);  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_72: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.view.default(permute_40, [8, 1, 14, 1, 14, 512]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_41: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.permute.default(view_72, [0, 1, 3, 2, 4, 5]);  view_72 = None
    view_73: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(permute_41, [8, 1, -1, 512]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add_34: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(view_73, arg22_1);  view_73 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_34, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 1, 196, 1]" = var_mean_10[0]
    getitem_37: "f32[8, 1, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_10: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_14: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_34, getitem_37);  getitem_37 = None
    mul_40: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = rsqrt_10 = None
    mul_41: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_40, arg23_1);  mul_40 = arg23_1 = None
    add_36: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_41, arg24_1);  mul_41 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_74: "f32[1568, 512]" = torch.ops.aten.view.default(add_36, [1568, 512]);  add_36 = None
    permute_42: "f32[512, 1536]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg144_1, view_74, permute_42);  arg144_1 = view_74 = permute_42 = None
    view_75: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 1, 196, 1536]);  addmm_16 = None
    view_76: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_75, [8, 1, 196, 3, 16, 32]);  view_75 = None
    permute_43: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_76, [3, 0, 4, 1, 2, 5]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_43);  permute_43 = None
    getitem_38: "f32[8, 16, 1, 196, 32]" = unbind_4[0]
    getitem_39: "f32[8, 16, 1, 196, 32]" = unbind_4[1]
    getitem_40: "f32[8, 16, 1, 196, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_42: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_38, 0.42044820762685725);  getitem_38 = None
    permute_44: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_39, [0, 1, 2, 4, 3]);  getitem_39 = None
    mul_43: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_44, 0.42044820762685725);  permute_44 = None
    expand_16: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_42, [8, 16, 1, 196, 32]);  mul_42 = None
    clone_34: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_77: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_34, [128, 196, 32]);  clone_34 = None
    expand_17: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_43, [8, 16, 1, 32, 196]);  mul_43 = None
    clone_35: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_78: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_35, [128, 32, 196]);  clone_35 = None
    bmm_8: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
    view_79: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_8, [8, 16, 1, 196, 196]);  bmm_8 = None
    amax_4: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_79, [-1], True)
    sub_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_79, amax_4);  view_79 = amax_4 = None
    exp_4: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_5: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    expand_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_4, [8, 16, 1, 196, 196]);  div_4 = None
    view_80: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_18, [128, 196, 196]);  expand_18 = None
    expand_19: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_40, [8, 16, 1, 196, 32]);  getitem_40 = None
    clone_36: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_81: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_36, [128, 196, 32]);  clone_36 = None
    bmm_9: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
    view_82: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_9, [8, 16, 1, 196, 32]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_45: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_82, [0, 2, 3, 4, 1]);  view_82 = None
    clone_37: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_83: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_37, [8, 1, 196, 512]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_84: "f32[1568, 512]" = torch.ops.aten.view.default(view_83, [1568, 512]);  view_83 = None
    permute_46: "f32[512, 512]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    addmm_17: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg146_1, view_84, permute_46);  arg146_1 = view_84 = permute_46 = None
    view_85: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_17, [8, 1, 196, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_38: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_37: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_34, clone_38);  add_34 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [3], correction = 0, keepdim = True)
    getitem_41: "f32[8, 1, 196, 1]" = var_mean_11[0]
    getitem_42: "f32[8, 1, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-06);  getitem_41 = None
    rsqrt_11: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_16: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_37, getitem_42);  getitem_42 = None
    mul_44: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_11);  sub_16 = rsqrt_11 = None
    mul_45: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_44, arg25_1);  mul_44 = arg25_1 = None
    add_39: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_45, arg26_1);  mul_45 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_86: "f32[1568, 512]" = torch.ops.aten.view.default(add_39, [1568, 512]);  add_39 = None
    permute_47: "f32[512, 2048]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_18: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg148_1, view_86, permute_47);  arg148_1 = view_86 = permute_47 = None
    view_87: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 1, 196, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_47: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_4: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_48: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_46, add_40);  mul_46 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_88: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_39, [1568, 2048]);  clone_39 = None
    permute_48: "f32[2048, 512]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_19: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg150_1, view_88, permute_48);  arg150_1 = view_88 = permute_48 = None
    view_89: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_19, [8, 1, 196, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_89);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_41: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_37, clone_40);  add_37 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_41, [3], correction = 0, keepdim = True)
    getitem_43: "f32[8, 1, 196, 1]" = var_mean_12[0]
    getitem_44: "f32[8, 1, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-06);  getitem_43 = None
    rsqrt_12: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_17: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_41, getitem_44);  getitem_44 = None
    mul_49: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = rsqrt_12 = None
    mul_50: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_49, arg27_1);  mul_49 = arg27_1 = None
    add_43: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_50, arg28_1);  mul_50 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_90: "f32[1568, 512]" = torch.ops.aten.view.default(add_43, [1568, 512]);  add_43 = None
    permute_49: "f32[512, 1536]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_20: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg152_1, view_90, permute_49);  arg152_1 = view_90 = permute_49 = None
    view_91: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 1, 196, 1536]);  addmm_20 = None
    view_92: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_91, [8, 1, 196, 3, 16, 32]);  view_91 = None
    permute_50: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_92, [3, 0, 4, 1, 2, 5]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_45: "f32[8, 16, 1, 196, 32]" = unbind_5[0]
    getitem_46: "f32[8, 16, 1, 196, 32]" = unbind_5[1]
    getitem_47: "f32[8, 16, 1, 196, 32]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_51: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_45, 0.42044820762685725);  getitem_45 = None
    permute_51: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_46, [0, 1, 2, 4, 3]);  getitem_46 = None
    mul_52: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_51, 0.42044820762685725);  permute_51 = None
    expand_20: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_51, [8, 16, 1, 196, 32]);  mul_51 = None
    clone_41: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_93: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_41, [128, 196, 32]);  clone_41 = None
    expand_21: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_52, [8, 16, 1, 32, 196]);  mul_52 = None
    clone_42: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_94: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_42, [128, 32, 196]);  clone_42 = None
    bmm_10: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
    view_95: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_10, [8, 16, 1, 196, 196]);  bmm_10 = None
    amax_5: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_95, [-1], True)
    sub_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_95, amax_5);  view_95 = amax_5 = None
    exp_5: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_6: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    expand_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_5, [8, 16, 1, 196, 196]);  div_5 = None
    view_96: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_22, [128, 196, 196]);  expand_22 = None
    expand_23: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_47, [8, 16, 1, 196, 32]);  getitem_47 = None
    clone_43: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_97: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_43, [128, 196, 32]);  clone_43 = None
    bmm_11: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_96, view_97);  view_96 = view_97 = None
    view_98: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_11, [8, 16, 1, 196, 32]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_52: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_98, [0, 2, 3, 4, 1]);  view_98 = None
    clone_44: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_99: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_44, [8, 1, 196, 512]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_100: "f32[1568, 512]" = torch.ops.aten.view.default(view_99, [1568, 512]);  view_99 = None
    permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_21: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg154_1, view_100, permute_53);  arg154_1 = view_100 = permute_53 = None
    view_101: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_21, [8, 1, 196, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_45: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_101);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_44: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_41, clone_45);  add_41 = clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_44, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 1, 196, 1]" = var_mean_13[0]
    getitem_49: "f32[8, 1, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_13: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_19: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_44, getitem_49);  getitem_49 = None
    mul_53: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
    mul_54: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_53, arg29_1);  mul_53 = arg29_1 = None
    add_46: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_54, arg30_1);  mul_54 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_102: "f32[1568, 512]" = torch.ops.aten.view.default(add_46, [1568, 512]);  add_46 = None
    permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_22: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg156_1, view_102, permute_54);  arg156_1 = view_102 = permute_54 = None
    view_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 1, 196, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_55: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.5)
    mul_56: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476);  view_103 = None
    erf_5: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_47: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_57: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_55, add_47);  mul_55 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_46: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_104: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_46, [1568, 2048]);  clone_46 = None
    permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_23: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg158_1, view_104, permute_55);  arg158_1 = view_104 = permute_55 = None
    view_105: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_23, [8, 1, 196, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_47: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_48: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_44, clone_47);  add_44 = clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_48, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 1, 196, 1]" = var_mean_14[0]
    getitem_51: "f32[8, 1, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_14: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_20: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_48, getitem_51);  getitem_51 = None
    mul_58: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = rsqrt_14 = None
    mul_59: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_58, arg31_1);  mul_58 = arg31_1 = None
    add_50: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_59, arg32_1);  mul_59 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_106: "f32[1568, 512]" = torch.ops.aten.view.default(add_50, [1568, 512]);  add_50 = None
    permute_56: "f32[512, 1536]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_24: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg160_1, view_106, permute_56);  arg160_1 = view_106 = permute_56 = None
    view_107: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 1, 196, 1536]);  addmm_24 = None
    view_108: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_107, [8, 1, 196, 3, 16, 32]);  view_107 = None
    permute_57: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_108, [3, 0, 4, 1, 2, 5]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
    getitem_52: "f32[8, 16, 1, 196, 32]" = unbind_6[0]
    getitem_53: "f32[8, 16, 1, 196, 32]" = unbind_6[1]
    getitem_54: "f32[8, 16, 1, 196, 32]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_60: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_52, 0.42044820762685725);  getitem_52 = None
    permute_58: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_53, [0, 1, 2, 4, 3]);  getitem_53 = None
    mul_61: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_58, 0.42044820762685725);  permute_58 = None
    expand_24: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_60, [8, 16, 1, 196, 32]);  mul_60 = None
    clone_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_109: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_48, [128, 196, 32]);  clone_48 = None
    expand_25: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_61, [8, 16, 1, 32, 196]);  mul_61 = None
    clone_49: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_110: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_49, [128, 32, 196]);  clone_49 = None
    bmm_12: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_109, view_110);  view_109 = view_110 = None
    view_111: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_12, [8, 16, 1, 196, 196]);  bmm_12 = None
    amax_6: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_111, [-1], True)
    sub_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_111, amax_6);  view_111 = amax_6 = None
    exp_6: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_7: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    expand_26: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_6, [8, 16, 1, 196, 196]);  div_6 = None
    view_112: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_26, [128, 196, 196]);  expand_26 = None
    expand_27: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_54, [8, 16, 1, 196, 32]);  getitem_54 = None
    clone_50: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_113: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_50, [128, 196, 32]);  clone_50 = None
    bmm_13: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_112, view_113);  view_112 = view_113 = None
    view_114: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_13, [8, 16, 1, 196, 32]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_59: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_114, [0, 2, 3, 4, 1]);  view_114 = None
    clone_51: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    view_115: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_51, [8, 1, 196, 512]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_116: "f32[1568, 512]" = torch.ops.aten.view.default(view_115, [1568, 512]);  view_115 = None
    permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    addmm_25: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg162_1, view_116, permute_60);  arg162_1 = view_116 = permute_60 = None
    view_117: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_25, [8, 1, 196, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_52: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_117);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_51: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_48, clone_52);  add_48 = clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_51, [3], correction = 0, keepdim = True)
    getitem_55: "f32[8, 1, 196, 1]" = var_mean_15[0]
    getitem_56: "f32[8, 1, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_15: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_22: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_51, getitem_56);  getitem_56 = None
    mul_62: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_15);  sub_22 = rsqrt_15 = None
    mul_63: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_62, arg33_1);  mul_62 = arg33_1 = None
    add_53: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_63, arg34_1);  mul_63 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[1568, 512]" = torch.ops.aten.view.default(add_53, [1568, 512]);  add_53 = None
    permute_61: "f32[512, 2048]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_26: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg164_1, view_118, permute_61);  arg164_1 = view_118 = permute_61 = None
    view_119: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 1, 196, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_64: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.5)
    mul_65: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.7071067811865476);  view_119 = None
    erf_6: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_54: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_66: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_64, add_54);  mul_64 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_120: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_53, [1568, 2048]);  clone_53 = None
    permute_62: "f32[2048, 512]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_27: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg166_1, view_120, permute_62);  arg166_1 = view_120 = permute_62 = None
    view_121: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_27, [8, 1, 196, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_121);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_55: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_51, clone_54);  add_51 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_55, [3], correction = 0, keepdim = True)
    getitem_57: "f32[8, 1, 196, 1]" = var_mean_16[0]
    getitem_58: "f32[8, 1, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-06);  getitem_57 = None
    rsqrt_16: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_23: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_55, getitem_58);  getitem_58 = None
    mul_67: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = rsqrt_16 = None
    mul_68: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_67, arg35_1);  mul_67 = arg35_1 = None
    add_57: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_68, arg36_1);  mul_68 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_122: "f32[1568, 512]" = torch.ops.aten.view.default(add_57, [1568, 512]);  add_57 = None
    permute_63: "f32[512, 1536]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg168_1, view_122, permute_63);  arg168_1 = view_122 = permute_63 = None
    view_123: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 1, 196, 1536]);  addmm_28 = None
    view_124: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_123, [8, 1, 196, 3, 16, 32]);  view_123 = None
    permute_64: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_124, [3, 0, 4, 1, 2, 5]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_64);  permute_64 = None
    getitem_59: "f32[8, 16, 1, 196, 32]" = unbind_7[0]
    getitem_60: "f32[8, 16, 1, 196, 32]" = unbind_7[1]
    getitem_61: "f32[8, 16, 1, 196, 32]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_69: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_59, 0.42044820762685725);  getitem_59 = None
    permute_65: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_60, [0, 1, 2, 4, 3]);  getitem_60 = None
    mul_70: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_65, 0.42044820762685725);  permute_65 = None
    expand_28: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_69, [8, 16, 1, 196, 32]);  mul_69 = None
    clone_55: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_125: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_55, [128, 196, 32]);  clone_55 = None
    expand_29: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_70, [8, 16, 1, 32, 196]);  mul_70 = None
    clone_56: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_126: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_56, [128, 32, 196]);  clone_56 = None
    bmm_14: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_125, view_126);  view_125 = view_126 = None
    view_127: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_14, [8, 16, 1, 196, 196]);  bmm_14 = None
    amax_7: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_127, [-1], True)
    sub_24: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_127, amax_7);  view_127 = amax_7 = None
    exp_7: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_8: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    expand_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_7, [8, 16, 1, 196, 196]);  div_7 = None
    view_128: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_30, [128, 196, 196]);  expand_30 = None
    expand_31: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_61, [8, 16, 1, 196, 32]);  getitem_61 = None
    clone_57: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_129: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_57, [128, 196, 32]);  clone_57 = None
    bmm_15: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_128, view_129);  view_128 = view_129 = None
    view_130: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_15, [8, 16, 1, 196, 32]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_66: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_130, [0, 2, 3, 4, 1]);  view_130 = None
    clone_58: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_131: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_58, [8, 1, 196, 512]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_132: "f32[1568, 512]" = torch.ops.aten.view.default(view_131, [1568, 512]);  view_131 = None
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_29: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg170_1, view_132, permute_67);  arg170_1 = view_132 = permute_67 = None
    view_133: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_29, [8, 1, 196, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_59: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_58: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_55, clone_59);  add_55 = clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_58, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 1, 196, 1]" = var_mean_17[0]
    getitem_63: "f32[8, 1, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_17: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_25: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_58, getitem_63);  getitem_63 = None
    mul_71: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_17);  sub_25 = rsqrt_17 = None
    mul_72: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_71, arg37_1);  mul_71 = arg37_1 = None
    add_60: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_72, arg38_1);  mul_72 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_134: "f32[1568, 512]" = torch.ops.aten.view.default(add_60, [1568, 512]);  add_60 = None
    permute_68: "f32[512, 2048]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_30: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg172_1, view_134, permute_68);  arg172_1 = view_134 = permute_68 = None
    view_135: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 1, 196, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_73: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
    mul_74: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476);  view_135 = None
    erf_7: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_61: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_75: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_73, add_61);  mul_73 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_60: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_75);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_136: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_60, [1568, 2048]);  clone_60 = None
    permute_69: "f32[2048, 512]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_31: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg174_1, view_136, permute_69);  arg174_1 = view_136 = permute_69 = None
    view_137: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_31, [8, 1, 196, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_61: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_137);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_62: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_58, clone_61);  add_58 = clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_62, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 1, 196, 1]" = var_mean_18[0]
    getitem_65: "f32[8, 1, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_18: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_26: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_62, getitem_65);  getitem_65 = None
    mul_76: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = rsqrt_18 = None
    mul_77: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_76, arg39_1);  mul_76 = arg39_1 = None
    add_64: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_77, arg40_1);  mul_77 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_138: "f32[1568, 512]" = torch.ops.aten.view.default(add_64, [1568, 512]);  add_64 = None
    permute_70: "f32[512, 1536]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_32: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg176_1, view_138, permute_70);  arg176_1 = view_138 = permute_70 = None
    view_139: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 1, 196, 1536]);  addmm_32 = None
    view_140: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_139, [8, 1, 196, 3, 16, 32]);  view_139 = None
    permute_71: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_140, [3, 0, 4, 1, 2, 5]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_71);  permute_71 = None
    getitem_66: "f32[8, 16, 1, 196, 32]" = unbind_8[0]
    getitem_67: "f32[8, 16, 1, 196, 32]" = unbind_8[1]
    getitem_68: "f32[8, 16, 1, 196, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_78: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_66, 0.42044820762685725);  getitem_66 = None
    permute_72: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_67, [0, 1, 2, 4, 3]);  getitem_67 = None
    mul_79: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_72, 0.42044820762685725);  permute_72 = None
    expand_32: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_78, [8, 16, 1, 196, 32]);  mul_78 = None
    clone_62: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_141: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_62, [128, 196, 32]);  clone_62 = None
    expand_33: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_79, [8, 16, 1, 32, 196]);  mul_79 = None
    clone_63: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_142: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_63, [128, 32, 196]);  clone_63 = None
    bmm_16: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
    view_143: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_16, [8, 16, 1, 196, 196]);  bmm_16 = None
    amax_8: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_143, [-1], True)
    sub_27: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_143, amax_8);  view_143 = amax_8 = None
    exp_8: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_9: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    expand_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_8, [8, 16, 1, 196, 196]);  div_8 = None
    view_144: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_34, [128, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_68, [8, 16, 1, 196, 32]);  getitem_68 = None
    clone_64: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_145: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_64, [128, 196, 32]);  clone_64 = None
    bmm_17: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
    view_146: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_17, [8, 16, 1, 196, 32]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_73: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_146, [0, 2, 3, 4, 1]);  view_146 = None
    clone_65: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_147: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_65, [8, 1, 196, 512]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_148: "f32[1568, 512]" = torch.ops.aten.view.default(view_147, [1568, 512]);  view_147 = None
    permute_74: "f32[512, 512]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    addmm_33: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg178_1, view_148, permute_74);  arg178_1 = view_148 = permute_74 = None
    view_149: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_33, [8, 1, 196, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_66: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_149);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_65: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_62, clone_66);  add_62 = clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_65, [3], correction = 0, keepdim = True)
    getitem_69: "f32[8, 1, 196, 1]" = var_mean_19[0]
    getitem_70: "f32[8, 1, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-06);  getitem_69 = None
    rsqrt_19: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_28: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_65, getitem_70);  getitem_70 = None
    mul_80: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_19);  sub_28 = rsqrt_19 = None
    mul_81: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_80, arg41_1);  mul_80 = arg41_1 = None
    add_67: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_81, arg42_1);  mul_81 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_150: "f32[1568, 512]" = torch.ops.aten.view.default(add_67, [1568, 512]);  add_67 = None
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_34: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg180_1, view_150, permute_75);  arg180_1 = view_150 = permute_75 = None
    view_151: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 1, 196, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_83: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_8: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_68: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_84: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_82, add_68);  mul_82 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_67: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_67, [1568, 2048]);  clone_67 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_35: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg182_1, view_152, permute_76);  arg182_1 = view_152 = permute_76 = None
    view_153: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_35, [8, 1, 196, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_68: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_69: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_65, clone_68);  add_65 = clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_69, [3], correction = 0, keepdim = True)
    getitem_71: "f32[8, 1, 196, 1]" = var_mean_20[0]
    getitem_72: "f32[8, 1, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-06);  getitem_71 = None
    rsqrt_20: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_29: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_69, getitem_72);  getitem_72 = None
    mul_85: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = rsqrt_20 = None
    mul_86: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_85, arg43_1);  mul_85 = arg43_1 = None
    add_71: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_86, arg44_1);  mul_86 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_154: "f32[1568, 512]" = torch.ops.aten.view.default(add_71, [1568, 512]);  add_71 = None
    permute_77: "f32[512, 1536]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_36: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg184_1, view_154, permute_77);  arg184_1 = view_154 = permute_77 = None
    view_155: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_36, [8, 1, 196, 1536]);  addmm_36 = None
    view_156: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_155, [8, 1, 196, 3, 16, 32]);  view_155 = None
    permute_78: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_156, [3, 0, 4, 1, 2, 5]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_78);  permute_78 = None
    getitem_73: "f32[8, 16, 1, 196, 32]" = unbind_9[0]
    getitem_74: "f32[8, 16, 1, 196, 32]" = unbind_9[1]
    getitem_75: "f32[8, 16, 1, 196, 32]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_87: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_73, 0.42044820762685725);  getitem_73 = None
    permute_79: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_74, [0, 1, 2, 4, 3]);  getitem_74 = None
    mul_88: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_79, 0.42044820762685725);  permute_79 = None
    expand_36: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_87, [8, 16, 1, 196, 32]);  mul_87 = None
    clone_69: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_157: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_69, [128, 196, 32]);  clone_69 = None
    expand_37: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_88, [8, 16, 1, 32, 196]);  mul_88 = None
    clone_70: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_158: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_70, [128, 32, 196]);  clone_70 = None
    bmm_18: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
    view_159: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_18, [8, 16, 1, 196, 196]);  bmm_18 = None
    amax_9: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_159, [-1], True)
    sub_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_159, amax_9);  view_159 = amax_9 = None
    exp_9: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_10: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    expand_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_9, [8, 16, 1, 196, 196]);  div_9 = None
    view_160: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_38, [128, 196, 196]);  expand_38 = None
    expand_39: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_75, [8, 16, 1, 196, 32]);  getitem_75 = None
    clone_71: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_161: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_71, [128, 196, 32]);  clone_71 = None
    bmm_19: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
    view_162: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_19, [8, 16, 1, 196, 32]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_80: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_162, [0, 2, 3, 4, 1]);  view_162 = None
    clone_72: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_163: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_72, [8, 1, 196, 512]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_164: "f32[1568, 512]" = torch.ops.aten.view.default(view_163, [1568, 512]);  view_163 = None
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_37: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg186_1, view_164, permute_81);  arg186_1 = view_164 = permute_81 = None
    view_165: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_37, [8, 1, 196, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_73: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_165);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_72: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_69, clone_73);  add_69 = clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_72, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 1, 196, 1]" = var_mean_21[0]
    getitem_77: "f32[8, 1, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_21: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_31: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_72, getitem_77);  getitem_77 = None
    mul_89: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_21);  sub_31 = rsqrt_21 = None
    mul_90: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_89, arg45_1);  mul_89 = arg45_1 = None
    add_74: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_90, arg46_1);  mul_90 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_166: "f32[1568, 512]" = torch.ops.aten.view.default(add_74, [1568, 512]);  add_74 = None
    permute_82: "f32[512, 2048]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_38: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg188_1, view_166, permute_82);  arg188_1 = view_166 = permute_82 = None
    view_167: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 1, 196, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_91: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    mul_92: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476);  view_167 = None
    erf_9: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
    add_75: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_93: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_91, add_75);  mul_91 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_74: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_93);  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_74, [1568, 2048]);  clone_74 = None
    permute_83: "f32[2048, 512]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_39: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg190_1, view_168, permute_83);  arg190_1 = view_168 = permute_83 = None
    view_169: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_39, [8, 1, 196, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_75: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_169);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_76: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_72, clone_75);  add_72 = clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_76, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 1, 196, 1]" = var_mean_22[0]
    getitem_79: "f32[8, 1, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_22: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_32: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_76, getitem_79);  getitem_79 = None
    mul_94: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = rsqrt_22 = None
    mul_95: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_94, arg47_1);  mul_94 = arg47_1 = None
    add_78: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_95, arg48_1);  mul_95 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_170: "f32[1568, 512]" = torch.ops.aten.view.default(add_78, [1568, 512]);  add_78 = None
    permute_84: "f32[512, 1536]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg192_1, view_170, permute_84);  arg192_1 = view_170 = permute_84 = None
    view_171: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 1, 196, 1536]);  addmm_40 = None
    view_172: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_171, [8, 1, 196, 3, 16, 32]);  view_171 = None
    permute_85: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_172, [3, 0, 4, 1, 2, 5]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_85);  permute_85 = None
    getitem_80: "f32[8, 16, 1, 196, 32]" = unbind_10[0]
    getitem_81: "f32[8, 16, 1, 196, 32]" = unbind_10[1]
    getitem_82: "f32[8, 16, 1, 196, 32]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_96: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_80, 0.42044820762685725);  getitem_80 = None
    permute_86: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_81, [0, 1, 2, 4, 3]);  getitem_81 = None
    mul_97: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_86, 0.42044820762685725);  permute_86 = None
    expand_40: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_96, [8, 16, 1, 196, 32]);  mul_96 = None
    clone_76: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_173: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_76, [128, 196, 32]);  clone_76 = None
    expand_41: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_97, [8, 16, 1, 32, 196]);  mul_97 = None
    clone_77: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_174: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_77, [128, 32, 196]);  clone_77 = None
    bmm_20: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_173, view_174);  view_173 = view_174 = None
    view_175: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_20, [8, 16, 1, 196, 196]);  bmm_20 = None
    amax_10: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_175, [-1], True)
    sub_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_175, amax_10);  view_175 = amax_10 = None
    exp_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_11: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    expand_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_10, [8, 16, 1, 196, 196]);  div_10 = None
    view_176: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_42, [128, 196, 196]);  expand_42 = None
    expand_43: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_82, [8, 16, 1, 196, 32]);  getitem_82 = None
    clone_78: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_177: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_78, [128, 196, 32]);  clone_78 = None
    bmm_21: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_176, view_177);  view_176 = view_177 = None
    view_178: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_21, [8, 16, 1, 196, 32]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_87: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_178, [0, 2, 3, 4, 1]);  view_178 = None
    clone_79: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_179: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_79, [8, 1, 196, 512]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_180: "f32[1568, 512]" = torch.ops.aten.view.default(view_179, [1568, 512]);  view_179 = None
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    addmm_41: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg194_1, view_180, permute_88);  arg194_1 = view_180 = permute_88 = None
    view_181: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_41, [8, 1, 196, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_80: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_181);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_79: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_76, clone_80);  add_76 = clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_79, [3], correction = 0, keepdim = True)
    getitem_83: "f32[8, 1, 196, 1]" = var_mean_23[0]
    getitem_84: "f32[8, 1, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-06);  getitem_83 = None
    rsqrt_23: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_34: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_84);  getitem_84 = None
    mul_98: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_23);  sub_34 = rsqrt_23 = None
    mul_99: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_98, arg49_1);  mul_98 = arg49_1 = None
    add_81: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_99, arg50_1);  mul_99 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_182: "f32[1568, 512]" = torch.ops.aten.view.default(add_81, [1568, 512]);  add_81 = None
    permute_89: "f32[512, 2048]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_42: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg196_1, view_182, permute_89);  arg196_1 = view_182 = permute_89 = None
    view_183: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 1, 196, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_100: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.5)
    mul_101: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476);  view_183 = None
    erf_10: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_82: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_102: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_100, add_82);  mul_100 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_81: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_102);  mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_184: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_81, [1568, 2048]);  clone_81 = None
    permute_90: "f32[2048, 512]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_43: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg198_1, view_184, permute_90);  arg198_1 = view_184 = permute_90 = None
    view_185: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_43, [8, 1, 196, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_82: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_185);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_83: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_79, clone_82);  add_79 = clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_83, [3], correction = 0, keepdim = True)
    getitem_85: "f32[8, 1, 196, 1]" = var_mean_24[0]
    getitem_86: "f32[8, 1, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_85, 1e-06);  getitem_85 = None
    rsqrt_24: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_35: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_83, getitem_86);  getitem_86 = None
    mul_103: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = rsqrt_24 = None
    mul_104: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_103, arg51_1);  mul_103 = arg51_1 = None
    add_85: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_104, arg52_1);  mul_104 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_186: "f32[1568, 512]" = torch.ops.aten.view.default(add_85, [1568, 512]);  add_85 = None
    permute_91: "f32[512, 1536]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    addmm_44: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg200_1, view_186, permute_91);  arg200_1 = view_186 = permute_91 = None
    view_187: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_44, [8, 1, 196, 1536]);  addmm_44 = None
    view_188: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_187, [8, 1, 196, 3, 16, 32]);  view_187 = None
    permute_92: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_188, [3, 0, 4, 1, 2, 5]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_92);  permute_92 = None
    getitem_87: "f32[8, 16, 1, 196, 32]" = unbind_11[0]
    getitem_88: "f32[8, 16, 1, 196, 32]" = unbind_11[1]
    getitem_89: "f32[8, 16, 1, 196, 32]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_105: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_87, 0.42044820762685725);  getitem_87 = None
    permute_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_88, [0, 1, 2, 4, 3]);  getitem_88 = None
    mul_106: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_93, 0.42044820762685725);  permute_93 = None
    expand_44: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_105, [8, 16, 1, 196, 32]);  mul_105 = None
    clone_83: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_189: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_83, [128, 196, 32]);  clone_83 = None
    expand_45: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_106, [8, 16, 1, 32, 196]);  mul_106 = None
    clone_84: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_190: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_84, [128, 32, 196]);  clone_84 = None
    bmm_22: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_189, view_190);  view_189 = view_190 = None
    view_191: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_22, [8, 16, 1, 196, 196]);  bmm_22 = None
    amax_11: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_191, [-1], True)
    sub_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_191, amax_11);  view_191 = amax_11 = None
    exp_11: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_12: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    expand_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_11, [8, 16, 1, 196, 196]);  div_11 = None
    view_192: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_46, [128, 196, 196]);  expand_46 = None
    expand_47: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_89, [8, 16, 1, 196, 32]);  getitem_89 = None
    clone_85: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_193: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_85, [128, 196, 32]);  clone_85 = None
    bmm_23: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_192, view_193);  view_192 = view_193 = None
    view_194: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_23, [8, 16, 1, 196, 32]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_94: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_194, [0, 2, 3, 4, 1]);  view_194 = None
    clone_86: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    view_195: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_86, [8, 1, 196, 512]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_196: "f32[1568, 512]" = torch.ops.aten.view.default(view_195, [1568, 512]);  view_195 = None
    permute_95: "f32[512, 512]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    addmm_45: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg202_1, view_196, permute_95);  arg202_1 = view_196 = permute_95 = None
    view_197: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_45, [8, 1, 196, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_87: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_197);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_86: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_83, clone_87);  add_83 = clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_86, [3], correction = 0, keepdim = True)
    getitem_90: "f32[8, 1, 196, 1]" = var_mean_25[0]
    getitem_91: "f32[8, 1, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_25: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_37: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_86, getitem_91);  getitem_91 = None
    mul_107: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
    mul_108: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_107, arg53_1);  mul_107 = arg53_1 = None
    add_88: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_108, arg54_1);  mul_108 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[1568, 512]" = torch.ops.aten.view.default(add_88, [1568, 512]);  add_88 = None
    permute_96: "f32[512, 2048]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    addmm_46: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg204_1, view_198, permute_96);  arg204_1 = view_198 = permute_96 = None
    view_199: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 1, 196, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_109: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
    mul_110: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476);  view_199 = None
    erf_11: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_89: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_111: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_109, add_89);  mul_109 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_88: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_111);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_200: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_88, [1568, 2048]);  clone_88 = None
    permute_97: "f32[2048, 512]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    addmm_47: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg206_1, view_200, permute_97);  arg206_1 = view_200 = permute_97 = None
    view_201: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_47, [8, 1, 196, 512]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_89: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_201);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_90: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_86, clone_89);  add_86 = clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_90, [3], correction = 0, keepdim = True)
    getitem_92: "f32[8, 1, 196, 1]" = var_mean_26[0]
    getitem_93: "f32[8, 1, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_26: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_38: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_90, getitem_93);  getitem_93 = None
    mul_112: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = rsqrt_26 = None
    mul_113: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_112, arg55_1);  mul_112 = arg55_1 = None
    add_92: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_113, arg56_1);  mul_113 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_202: "f32[1568, 512]" = torch.ops.aten.view.default(add_92, [1568, 512]);  add_92 = None
    permute_98: "f32[512, 1536]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
    addmm_48: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg208_1, view_202, permute_98);  arg208_1 = view_202 = permute_98 = None
    view_203: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_48, [8, 1, 196, 1536]);  addmm_48 = None
    view_204: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_203, [8, 1, 196, 3, 16, 32]);  view_203 = None
    permute_99: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_204, [3, 0, 4, 1, 2, 5]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_99);  permute_99 = None
    getitem_94: "f32[8, 16, 1, 196, 32]" = unbind_12[0]
    getitem_95: "f32[8, 16, 1, 196, 32]" = unbind_12[1]
    getitem_96: "f32[8, 16, 1, 196, 32]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_114: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_94, 0.42044820762685725);  getitem_94 = None
    permute_100: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_95, [0, 1, 2, 4, 3]);  getitem_95 = None
    mul_115: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_100, 0.42044820762685725);  permute_100 = None
    expand_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_114, [8, 16, 1, 196, 32]);  mul_114 = None
    clone_90: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_205: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_90, [128, 196, 32]);  clone_90 = None
    expand_49: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_115, [8, 16, 1, 32, 196]);  mul_115 = None
    clone_91: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_206: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_91, [128, 32, 196]);  clone_91 = None
    bmm_24: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_205, view_206);  view_205 = view_206 = None
    view_207: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_24, [8, 16, 1, 196, 196]);  bmm_24 = None
    amax_12: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_207, [-1], True)
    sub_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_207, amax_12);  view_207 = amax_12 = None
    exp_12: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_13: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    expand_50: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_12, [8, 16, 1, 196, 196]);  div_12 = None
    view_208: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_50, [128, 196, 196]);  expand_50 = None
    expand_51: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_96, [8, 16, 1, 196, 32]);  getitem_96 = None
    clone_92: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_209: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_92, [128, 196, 32]);  clone_92 = None
    bmm_25: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_208, view_209);  view_208 = view_209 = None
    view_210: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_25, [8, 16, 1, 196, 32]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_101: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_210, [0, 2, 3, 4, 1]);  view_210 = None
    clone_93: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    view_211: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_93, [8, 1, 196, 512]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_212: "f32[1568, 512]" = torch.ops.aten.view.default(view_211, [1568, 512]);  view_211 = None
    permute_102: "f32[512, 512]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
    addmm_49: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg210_1, view_212, permute_102);  arg210_1 = view_212 = permute_102 = None
    view_213: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_49, [8, 1, 196, 512]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_94: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_213);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_93: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_90, clone_94);  add_90 = clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_93, [3], correction = 0, keepdim = True)
    getitem_97: "f32[8, 1, 196, 1]" = var_mean_27[0]
    getitem_98: "f32[8, 1, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_27: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_40: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_93, getitem_98);  getitem_98 = None
    mul_116: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_27);  sub_40 = rsqrt_27 = None
    mul_117: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_116, arg57_1);  mul_116 = arg57_1 = None
    add_95: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_117, arg58_1);  mul_117 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_214: "f32[1568, 512]" = torch.ops.aten.view.default(add_95, [1568, 512]);  add_95 = None
    permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
    addmm_50: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg212_1, view_214, permute_103);  arg212_1 = view_214 = permute_103 = None
    view_215: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 1, 196, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_118: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    mul_119: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.7071067811865476);  view_215 = None
    erf_12: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
    add_96: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_120: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_118, add_96);  mul_118 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_95: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_120);  mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_216: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_95, [1568, 2048]);  clone_95 = None
    permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    addmm_51: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg214_1, view_216, permute_104);  arg214_1 = view_216 = permute_104 = None
    view_217: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_51, [8, 1, 196, 512]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_97: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_93, clone_96);  add_93 = clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_97, [3], correction = 0, keepdim = True)
    getitem_99: "f32[8, 1, 196, 1]" = var_mean_28[0]
    getitem_100: "f32[8, 1, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_28: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_41: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_97, getitem_100);  getitem_100 = None
    mul_121: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = rsqrt_28 = None
    mul_122: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_121, arg59_1);  mul_121 = arg59_1 = None
    add_99: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_122, arg60_1);  mul_122 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_218: "f32[1568, 512]" = torch.ops.aten.view.default(add_99, [1568, 512]);  add_99 = None
    permute_105: "f32[512, 1536]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg216_1, view_218, permute_105);  arg216_1 = view_218 = permute_105 = None
    view_219: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 1, 196, 1536]);  addmm_52 = None
    view_220: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_219, [8, 1, 196, 3, 16, 32]);  view_219 = None
    permute_106: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_220, [3, 0, 4, 1, 2, 5]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_106);  permute_106 = None
    getitem_101: "f32[8, 16, 1, 196, 32]" = unbind_13[0]
    getitem_102: "f32[8, 16, 1, 196, 32]" = unbind_13[1]
    getitem_103: "f32[8, 16, 1, 196, 32]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_123: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_101, 0.42044820762685725);  getitem_101 = None
    permute_107: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_102, [0, 1, 2, 4, 3]);  getitem_102 = None
    mul_124: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_107, 0.42044820762685725);  permute_107 = None
    expand_52: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_123, [8, 16, 1, 196, 32]);  mul_123 = None
    clone_97: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_221: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_97, [128, 196, 32]);  clone_97 = None
    expand_53: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_124, [8, 16, 1, 32, 196]);  mul_124 = None
    clone_98: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_222: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_98, [128, 32, 196]);  clone_98 = None
    bmm_26: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_221, view_222);  view_221 = view_222 = None
    view_223: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_26, [8, 16, 1, 196, 196]);  bmm_26 = None
    amax_13: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_223, amax_13);  view_223 = amax_13 = None
    exp_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_14: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    expand_54: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_13, [8, 16, 1, 196, 196]);  div_13 = None
    view_224: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_54, [128, 196, 196]);  expand_54 = None
    expand_55: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_103, [8, 16, 1, 196, 32]);  getitem_103 = None
    clone_99: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_225: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_99, [128, 196, 32]);  clone_99 = None
    bmm_27: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
    view_226: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_27, [8, 16, 1, 196, 32]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_108: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_226, [0, 2, 3, 4, 1]);  view_226 = None
    clone_100: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    view_227: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_100, [8, 1, 196, 512]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_228: "f32[1568, 512]" = torch.ops.aten.view.default(view_227, [1568, 512]);  view_227 = None
    permute_109: "f32[512, 512]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    addmm_53: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg218_1, view_228, permute_109);  arg218_1 = view_228 = permute_109 = None
    view_229: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_53, [8, 1, 196, 512]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_101: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_229);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_100: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_97, clone_101);  add_97 = clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_100, [3], correction = 0, keepdim = True)
    getitem_104: "f32[8, 1, 196, 1]" = var_mean_29[0]
    getitem_105: "f32[8, 1, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_29: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_43: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_100, getitem_105);  getitem_105 = None
    mul_125: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_29);  sub_43 = rsqrt_29 = None
    mul_126: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_125, arg61_1);  mul_125 = arg61_1 = None
    add_102: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_126, arg62_1);  mul_126 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_230: "f32[1568, 512]" = torch.ops.aten.view.default(add_102, [1568, 512]);  add_102 = None
    permute_110: "f32[512, 2048]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    addmm_54: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg220_1, view_230, permute_110);  arg220_1 = view_230 = permute_110 = None
    view_231: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 1, 196, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_127: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.5)
    mul_128: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.7071067811865476);  view_231 = None
    erf_13: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_129: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_127, add_103);  mul_127 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_102: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_129);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_232: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_102, [1568, 2048]);  clone_102 = None
    permute_111: "f32[2048, 512]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
    addmm_55: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg222_1, view_232, permute_111);  arg222_1 = view_232 = permute_111 = None
    view_233: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_55, [8, 1, 196, 512]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_103: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_233);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_104: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_100, clone_103);  add_100 = clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_104, [3], correction = 0, keepdim = True)
    getitem_106: "f32[8, 1, 196, 1]" = var_mean_30[0]
    getitem_107: "f32[8, 1, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_30: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_44: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_104, getitem_107);  getitem_107 = None
    mul_130: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = rsqrt_30 = None
    mul_131: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_130, arg63_1);  mul_130 = arg63_1 = None
    add_106: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_131, arg64_1);  mul_131 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_234: "f32[1568, 512]" = torch.ops.aten.view.default(add_106, [1568, 512]);  add_106 = None
    permute_112: "f32[512, 1536]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
    addmm_56: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg224_1, view_234, permute_112);  arg224_1 = view_234 = permute_112 = None
    view_235: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_56, [8, 1, 196, 1536]);  addmm_56 = None
    view_236: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_235, [8, 1, 196, 3, 16, 32]);  view_235 = None
    permute_113: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_236, [3, 0, 4, 1, 2, 5]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
    getitem_108: "f32[8, 16, 1, 196, 32]" = unbind_14[0]
    getitem_109: "f32[8, 16, 1, 196, 32]" = unbind_14[1]
    getitem_110: "f32[8, 16, 1, 196, 32]" = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_132: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_108, 0.42044820762685725);  getitem_108 = None
    permute_114: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_109, [0, 1, 2, 4, 3]);  getitem_109 = None
    mul_133: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_114, 0.42044820762685725);  permute_114 = None
    expand_56: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_132, [8, 16, 1, 196, 32]);  mul_132 = None
    clone_104: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_237: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_104, [128, 196, 32]);  clone_104 = None
    expand_57: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_133, [8, 16, 1, 32, 196]);  mul_133 = None
    clone_105: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_238: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_105, [128, 32, 196]);  clone_105 = None
    bmm_28: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_237, view_238);  view_237 = view_238 = None
    view_239: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_28, [8, 16, 1, 196, 196]);  bmm_28 = None
    amax_14: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_239, [-1], True)
    sub_45: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_239, amax_14);  view_239 = amax_14 = None
    exp_14: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_15: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    expand_58: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_14, [8, 16, 1, 196, 196]);  div_14 = None
    view_240: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_58, [128, 196, 196]);  expand_58 = None
    expand_59: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_110, [8, 16, 1, 196, 32]);  getitem_110 = None
    clone_106: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_241: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_106, [128, 196, 32]);  clone_106 = None
    bmm_29: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_240, view_241);  view_240 = view_241 = None
    view_242: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_29, [8, 16, 1, 196, 32]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_115: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_242, [0, 2, 3, 4, 1]);  view_242 = None
    clone_107: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_243: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_107, [8, 1, 196, 512]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_244: "f32[1568, 512]" = torch.ops.aten.view.default(view_243, [1568, 512]);  view_243 = None
    permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    addmm_57: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg226_1, view_244, permute_116);  arg226_1 = view_244 = permute_116 = None
    view_245: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_57, [8, 1, 196, 512]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_108: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_245);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_107: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_104, clone_108);  add_104 = clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_107, [3], correction = 0, keepdim = True)
    getitem_111: "f32[8, 1, 196, 1]" = var_mean_31[0]
    getitem_112: "f32[8, 1, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-06);  getitem_111 = None
    rsqrt_31: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_46: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_107, getitem_112);  getitem_112 = None
    mul_134: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_31);  sub_46 = rsqrt_31 = None
    mul_135: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_134, arg65_1);  mul_134 = arg65_1 = None
    add_109: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_135, arg66_1);  mul_135 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[1568, 512]" = torch.ops.aten.view.default(add_109, [1568, 512]);  add_109 = None
    permute_117: "f32[512, 2048]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    addmm_58: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg228_1, view_246, permute_117);  arg228_1 = view_246 = permute_117 = None
    view_247: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 1, 196, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_136: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.5)
    mul_137: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.7071067811865476);  view_247 = None
    erf_14: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_137);  mul_137 = None
    add_110: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_138: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_136, add_110);  mul_136 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_109: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_138);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_248: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_109, [1568, 2048]);  clone_109 = None
    permute_118: "f32[2048, 512]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
    addmm_59: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg230_1, view_248, permute_118);  arg230_1 = view_248 = permute_118 = None
    view_249: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_59, [8, 1, 196, 512]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_110: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_249);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_111: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_107, clone_110);  add_107 = clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_111, [3], correction = 0, keepdim = True)
    getitem_113: "f32[8, 1, 196, 1]" = var_mean_32[0]
    getitem_114: "f32[8, 1, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_113, 1e-06);  getitem_113 = None
    rsqrt_32: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_47: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_111, getitem_114);  getitem_114 = None
    mul_139: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = rsqrt_32 = None
    mul_140: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_139, arg67_1);  mul_139 = arg67_1 = None
    add_113: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_140, arg68_1);  mul_140 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_250: "f32[1568, 512]" = torch.ops.aten.view.default(add_113, [1568, 512]);  add_113 = None
    permute_119: "f32[512, 1536]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
    addmm_60: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg232_1, view_250, permute_119);  arg232_1 = view_250 = permute_119 = None
    view_251: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_60, [8, 1, 196, 1536]);  addmm_60 = None
    view_252: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_251, [8, 1, 196, 3, 16, 32]);  view_251 = None
    permute_120: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_252, [3, 0, 4, 1, 2, 5]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_120);  permute_120 = None
    getitem_115: "f32[8, 16, 1, 196, 32]" = unbind_15[0]
    getitem_116: "f32[8, 16, 1, 196, 32]" = unbind_15[1]
    getitem_117: "f32[8, 16, 1, 196, 32]" = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_141: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_115, 0.42044820762685725);  getitem_115 = None
    permute_121: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_116, [0, 1, 2, 4, 3]);  getitem_116 = None
    mul_142: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_121, 0.42044820762685725);  permute_121 = None
    expand_60: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_141, [8, 16, 1, 196, 32]);  mul_141 = None
    clone_111: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_253: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_111, [128, 196, 32]);  clone_111 = None
    expand_61: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_142, [8, 16, 1, 32, 196]);  mul_142 = None
    clone_112: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_254: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_112, [128, 32, 196]);  clone_112 = None
    bmm_30: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_253, view_254);  view_253 = view_254 = None
    view_255: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_30, [8, 16, 1, 196, 196]);  bmm_30 = None
    amax_15: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_255, [-1], True)
    sub_48: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_255, amax_15);  view_255 = amax_15 = None
    exp_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_16: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    expand_62: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_15, [8, 16, 1, 196, 196]);  div_15 = None
    view_256: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_62, [128, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_117, [8, 16, 1, 196, 32]);  getitem_117 = None
    clone_113: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_257: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_113, [128, 196, 32]);  clone_113 = None
    bmm_31: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
    view_258: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_31, [8, 16, 1, 196, 32]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_122: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_258, [0, 2, 3, 4, 1]);  view_258 = None
    clone_114: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_259: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_114, [8, 1, 196, 512]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_260: "f32[1568, 512]" = torch.ops.aten.view.default(view_259, [1568, 512]);  view_259 = None
    permute_123: "f32[512, 512]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
    addmm_61: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg234_1, view_260, permute_123);  arg234_1 = view_260 = permute_123 = None
    view_261: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_61, [8, 1, 196, 512]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_115: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_114: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_111, clone_115);  add_111 = clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_114, [3], correction = 0, keepdim = True)
    getitem_118: "f32[8, 1, 196, 1]" = var_mean_33[0]
    getitem_119: "f32[8, 1, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
    rsqrt_33: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_49: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_114, getitem_119);  getitem_119 = None
    mul_143: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_33);  sub_49 = rsqrt_33 = None
    mul_144: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_143, arg69_1);  mul_143 = arg69_1 = None
    add_116: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_144, arg70_1);  mul_144 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_262: "f32[1568, 512]" = torch.ops.aten.view.default(add_116, [1568, 512]);  add_116 = None
    permute_124: "f32[512, 2048]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
    addmm_62: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg236_1, view_262, permute_124);  arg236_1 = view_262 = permute_124 = None
    view_263: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 1, 196, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_145: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_146: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_15: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_117: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_147: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_145, add_117);  mul_145 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_116: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_147);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_264: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_116, [1568, 2048]);  clone_116 = None
    permute_125: "f32[2048, 512]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    addmm_63: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg238_1, view_264, permute_125);  arg238_1 = view_264 = permute_125 = None
    view_265: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_63, [8, 1, 196, 512]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_117: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_118: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_114, clone_117);  add_114 = clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_118, [3], correction = 0, keepdim = True)
    getitem_120: "f32[8, 1, 196, 1]" = var_mean_34[0]
    getitem_121: "f32[8, 1, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_34: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_50: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_118, getitem_121);  getitem_121 = None
    mul_148: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = rsqrt_34 = None
    mul_149: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_148, arg71_1);  mul_148 = arg71_1 = None
    add_120: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_149, arg72_1);  mul_149 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_266: "f32[1568, 512]" = torch.ops.aten.view.default(add_120, [1568, 512]);  add_120 = None
    permute_126: "f32[512, 1536]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    addmm_64: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg240_1, view_266, permute_126);  arg240_1 = view_266 = permute_126 = None
    view_267: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_64, [8, 1, 196, 1536]);  addmm_64 = None
    view_268: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_267, [8, 1, 196, 3, 16, 32]);  view_267 = None
    permute_127: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_268, [3, 0, 4, 1, 2, 5]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
    getitem_122: "f32[8, 16, 1, 196, 32]" = unbind_16[0]
    getitem_123: "f32[8, 16, 1, 196, 32]" = unbind_16[1]
    getitem_124: "f32[8, 16, 1, 196, 32]" = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_150: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_122, 0.42044820762685725);  getitem_122 = None
    permute_128: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_123, [0, 1, 2, 4, 3]);  getitem_123 = None
    mul_151: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_128, 0.42044820762685725);  permute_128 = None
    expand_64: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_150, [8, 16, 1, 196, 32]);  mul_150 = None
    clone_118: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_269: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_118, [128, 196, 32]);  clone_118 = None
    expand_65: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_151, [8, 16, 1, 32, 196]);  mul_151 = None
    clone_119: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_270: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_119, [128, 32, 196]);  clone_119 = None
    bmm_32: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_269, view_270);  view_269 = view_270 = None
    view_271: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_32, [8, 16, 1, 196, 196]);  bmm_32 = None
    amax_16: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_271, [-1], True)
    sub_51: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_271, amax_16);  view_271 = amax_16 = None
    exp_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_17: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    expand_66: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_16, [8, 16, 1, 196, 196]);  div_16 = None
    view_272: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_66, [128, 196, 196]);  expand_66 = None
    expand_67: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_124, [8, 16, 1, 196, 32]);  getitem_124 = None
    clone_120: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_273: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_120, [128, 196, 32]);  clone_120 = None
    bmm_33: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_272, view_273);  view_272 = view_273 = None
    view_274: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_33, [8, 16, 1, 196, 32]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_129: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_274, [0, 2, 3, 4, 1]);  view_274 = None
    clone_121: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_275: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_121, [8, 1, 196, 512]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_276: "f32[1568, 512]" = torch.ops.aten.view.default(view_275, [1568, 512]);  view_275 = None
    permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
    addmm_65: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg242_1, view_276, permute_130);  arg242_1 = view_276 = permute_130 = None
    view_277: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_65, [8, 1, 196, 512]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_122: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_277);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_121: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_118, clone_122);  add_118 = clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_121, [3], correction = 0, keepdim = True)
    getitem_125: "f32[8, 1, 196, 1]" = var_mean_35[0]
    getitem_126: "f32[8, 1, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-06);  getitem_125 = None
    rsqrt_35: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_52: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_121, getitem_126);  getitem_126 = None
    mul_152: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_35);  sub_52 = rsqrt_35 = None
    mul_153: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_152, arg73_1);  mul_152 = arg73_1 = None
    add_123: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_153, arg74_1);  mul_153 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_278: "f32[1568, 512]" = torch.ops.aten.view.default(add_123, [1568, 512]);  add_123 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    addmm_66: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg244_1, view_278, permute_131);  arg244_1 = view_278 = permute_131 = None
    view_279: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_66, [8, 1, 196, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
    mul_155: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476);  view_279 = None
    erf_16: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_124: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_156: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_154, add_124);  mul_154 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_123: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_280: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_123, [1568, 2048]);  clone_123 = None
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    addmm_67: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg246_1, view_280, permute_132);  arg246_1 = view_280 = permute_132 = None
    view_281: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_67, [8, 1, 196, 512]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_124: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_281);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_125: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_121, clone_124);  add_121 = clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_125, [3], correction = 0, keepdim = True)
    getitem_127: "f32[8, 1, 196, 1]" = var_mean_36[0]
    getitem_128: "f32[8, 1, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_127, 1e-06);  getitem_127 = None
    rsqrt_36: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_53: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_125, getitem_128);  getitem_128 = None
    mul_157: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = rsqrt_36 = None
    mul_158: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_157, arg75_1);  mul_157 = arg75_1 = None
    add_127: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_158, arg76_1);  mul_158 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_282: "f32[1568, 512]" = torch.ops.aten.view.default(add_127, [1568, 512]);  add_127 = None
    permute_133: "f32[512, 1536]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    addmm_68: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg248_1, view_282, permute_133);  arg248_1 = view_282 = permute_133 = None
    view_283: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_68, [8, 1, 196, 1536]);  addmm_68 = None
    view_284: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_283, [8, 1, 196, 3, 16, 32]);  view_283 = None
    permute_134: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_284, [3, 0, 4, 1, 2, 5]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_134);  permute_134 = None
    getitem_129: "f32[8, 16, 1, 196, 32]" = unbind_17[0]
    getitem_130: "f32[8, 16, 1, 196, 32]" = unbind_17[1]
    getitem_131: "f32[8, 16, 1, 196, 32]" = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_159: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_129, 0.42044820762685725);  getitem_129 = None
    permute_135: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_130, [0, 1, 2, 4, 3]);  getitem_130 = None
    mul_160: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_135, 0.42044820762685725);  permute_135 = None
    expand_68: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_159, [8, 16, 1, 196, 32]);  mul_159 = None
    clone_125: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_285: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_125, [128, 196, 32]);  clone_125 = None
    expand_69: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_160, [8, 16, 1, 32, 196]);  mul_160 = None
    clone_126: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_286: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_126, [128, 32, 196]);  clone_126 = None
    bmm_34: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_285, view_286);  view_285 = view_286 = None
    view_287: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_34, [8, 16, 1, 196, 196]);  bmm_34 = None
    amax_17: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_287, [-1], True)
    sub_54: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_287, amax_17);  view_287 = amax_17 = None
    exp_17: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_18: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    expand_70: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_17, [8, 16, 1, 196, 196]);  div_17 = None
    view_288: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_70, [128, 196, 196]);  expand_70 = None
    expand_71: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_131, [8, 16, 1, 196, 32]);  getitem_131 = None
    clone_127: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_289: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_127, [128, 196, 32]);  clone_127 = None
    bmm_35: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_288, view_289);  view_288 = view_289 = None
    view_290: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_35, [8, 16, 1, 196, 32]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_136: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_290, [0, 2, 3, 4, 1]);  view_290 = None
    clone_128: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    view_291: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_128, [8, 1, 196, 512]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_292: "f32[1568, 512]" = torch.ops.aten.view.default(view_291, [1568, 512]);  view_291 = None
    permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    addmm_69: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg250_1, view_292, permute_137);  arg250_1 = view_292 = permute_137 = None
    view_293: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_69, [8, 1, 196, 512]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_129: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_293);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_128: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_125, clone_129);  add_125 = clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_128, [3], correction = 0, keepdim = True)
    getitem_132: "f32[8, 1, 196, 1]" = var_mean_37[0]
    getitem_133: "f32[8, 1, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_37: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_55: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_128, getitem_133);  getitem_133 = None
    mul_161: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_37);  sub_55 = rsqrt_37 = None
    mul_162: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_161, arg77_1);  mul_161 = arg77_1 = None
    add_130: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_162, arg78_1);  mul_162 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_294: "f32[1568, 512]" = torch.ops.aten.view.default(add_130, [1568, 512]);  add_130 = None
    permute_138: "f32[512, 2048]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
    addmm_70: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg252_1, view_294, permute_138);  arg252_1 = view_294 = permute_138 = None
    view_295: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_70, [8, 1, 196, 2048]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_163: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.5)
    mul_164: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.7071067811865476);  view_295 = None
    erf_17: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_131: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_165: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_163, add_131);  mul_163 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_130: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_165);  mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_296: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_130, [1568, 2048]);  clone_130 = None
    permute_139: "f32[2048, 512]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    addmm_71: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg254_1, view_296, permute_139);  arg254_1 = view_296 = permute_139 = None
    view_297: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_71, [8, 1, 196, 512]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_131: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_297);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_132: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_128, clone_131);  add_128 = clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_132, [3], correction = 0, keepdim = True)
    getitem_134: "f32[8, 1, 196, 1]" = var_mean_38[0]
    getitem_135: "f32[8, 1, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
    rsqrt_38: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_56: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_132, getitem_135);  getitem_135 = None
    mul_166: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = rsqrt_38 = None
    mul_167: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_166, arg79_1);  mul_166 = arg79_1 = None
    add_134: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_167, arg80_1);  mul_167 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_298: "f32[1568, 512]" = torch.ops.aten.view.default(add_134, [1568, 512]);  add_134 = None
    permute_140: "f32[512, 1536]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
    addmm_72: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg256_1, view_298, permute_140);  arg256_1 = view_298 = permute_140 = None
    view_299: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_72, [8, 1, 196, 1536]);  addmm_72 = None
    view_300: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_299, [8, 1, 196, 3, 16, 32]);  view_299 = None
    permute_141: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_300, [3, 0, 4, 1, 2, 5]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_141);  permute_141 = None
    getitem_136: "f32[8, 16, 1, 196, 32]" = unbind_18[0]
    getitem_137: "f32[8, 16, 1, 196, 32]" = unbind_18[1]
    getitem_138: "f32[8, 16, 1, 196, 32]" = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_168: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_136, 0.42044820762685725);  getitem_136 = None
    permute_142: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_137, [0, 1, 2, 4, 3]);  getitem_137 = None
    mul_169: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_142, 0.42044820762685725);  permute_142 = None
    expand_72: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_168, [8, 16, 1, 196, 32]);  mul_168 = None
    clone_132: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_301: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_132, [128, 196, 32]);  clone_132 = None
    expand_73: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_169, [8, 16, 1, 32, 196]);  mul_169 = None
    clone_133: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_302: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_133, [128, 32, 196]);  clone_133 = None
    bmm_36: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_301, view_302);  view_301 = view_302 = None
    view_303: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_36, [8, 16, 1, 196, 196]);  bmm_36 = None
    amax_18: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_303, [-1], True)
    sub_57: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_303, amax_18);  view_303 = amax_18 = None
    exp_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_19: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    expand_74: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_18, [8, 16, 1, 196, 196]);  div_18 = None
    view_304: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_74, [128, 196, 196]);  expand_74 = None
    expand_75: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_138, [8, 16, 1, 196, 32]);  getitem_138 = None
    clone_134: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_305: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_134, [128, 196, 32]);  clone_134 = None
    bmm_37: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_304, view_305);  view_304 = view_305 = None
    view_306: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_37, [8, 16, 1, 196, 32]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_143: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_306, [0, 2, 3, 4, 1]);  view_306 = None
    clone_135: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    view_307: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_135, [8, 1, 196, 512]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_308: "f32[1568, 512]" = torch.ops.aten.view.default(view_307, [1568, 512]);  view_307 = None
    permute_144: "f32[512, 512]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
    addmm_73: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg258_1, view_308, permute_144);  arg258_1 = view_308 = permute_144 = None
    view_309: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_73, [8, 1, 196, 512]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_136: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_135: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_132, clone_136);  add_132 = clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_135, [3], correction = 0, keepdim = True)
    getitem_139: "f32[8, 1, 196, 1]" = var_mean_39[0]
    getitem_140: "f32[8, 1, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-06);  getitem_139 = None
    rsqrt_39: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_58: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_135, getitem_140);  getitem_140 = None
    mul_170: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_39);  sub_58 = rsqrt_39 = None
    mul_171: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_170, arg81_1);  mul_170 = arg81_1 = None
    add_137: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_171, arg82_1);  mul_171 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_310: "f32[1568, 512]" = torch.ops.aten.view.default(add_137, [1568, 512]);  add_137 = None
    permute_145: "f32[512, 2048]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    addmm_74: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg260_1, view_310, permute_145);  arg260_1 = view_310 = permute_145 = None
    view_311: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_74, [8, 1, 196, 2048]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_172: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.5)
    mul_173: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476);  view_311 = None
    erf_18: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_173);  mul_173 = None
    add_138: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_174: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_172, add_138);  mul_172 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_137: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_174);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_312: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_137, [1568, 2048]);  clone_137 = None
    permute_146: "f32[2048, 512]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    addmm_75: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg262_1, view_312, permute_146);  arg262_1 = view_312 = permute_146 = None
    view_313: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_75, [8, 1, 196, 512]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_138: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_313);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_139: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_135, clone_138);  add_135 = clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_139, [3], correction = 0, keepdim = True)
    getitem_141: "f32[8, 1, 196, 1]" = var_mean_40[0]
    getitem_142: "f32[8, 1, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-06);  getitem_141 = None
    rsqrt_40: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_59: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_139, getitem_142);  getitem_142 = None
    mul_175: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = rsqrt_40 = None
    mul_176: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_175, arg83_1);  mul_175 = arg83_1 = None
    add_141: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_176, arg84_1);  mul_176 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_314: "f32[1568, 512]" = torch.ops.aten.view.default(add_141, [1568, 512]);  add_141 = None
    permute_147: "f32[512, 1536]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
    addmm_76: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg264_1, view_314, permute_147);  arg264_1 = view_314 = permute_147 = None
    view_315: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_76, [8, 1, 196, 1536]);  addmm_76 = None
    view_316: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_315, [8, 1, 196, 3, 16, 32]);  view_315 = None
    permute_148: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_316, [3, 0, 4, 1, 2, 5]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_148);  permute_148 = None
    getitem_143: "f32[8, 16, 1, 196, 32]" = unbind_19[0]
    getitem_144: "f32[8, 16, 1, 196, 32]" = unbind_19[1]
    getitem_145: "f32[8, 16, 1, 196, 32]" = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_177: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_143, 0.42044820762685725);  getitem_143 = None
    permute_149: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_144, [0, 1, 2, 4, 3]);  getitem_144 = None
    mul_178: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_149, 0.42044820762685725);  permute_149 = None
    expand_76: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_177, [8, 16, 1, 196, 32]);  mul_177 = None
    clone_139: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_317: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_139, [128, 196, 32]);  clone_139 = None
    expand_77: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_178, [8, 16, 1, 32, 196]);  mul_178 = None
    clone_140: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_318: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_140, [128, 32, 196]);  clone_140 = None
    bmm_38: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_317, view_318);  view_317 = view_318 = None
    view_319: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_38, [8, 16, 1, 196, 196]);  bmm_38 = None
    amax_19: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_319, [-1], True)
    sub_60: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_319, amax_19);  view_319 = amax_19 = None
    exp_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_20: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    expand_78: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_19, [8, 16, 1, 196, 196]);  div_19 = None
    view_320: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_78, [128, 196, 196]);  expand_78 = None
    expand_79: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_145, [8, 16, 1, 196, 32]);  getitem_145 = None
    clone_141: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
    view_321: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_141, [128, 196, 32]);  clone_141 = None
    bmm_39: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
    view_322: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_39, [8, 16, 1, 196, 32]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_150: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_322, [0, 2, 3, 4, 1]);  view_322 = None
    clone_142: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_323: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_142, [8, 1, 196, 512]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_324: "f32[1568, 512]" = torch.ops.aten.view.default(view_323, [1568, 512]);  view_323 = None
    permute_151: "f32[512, 512]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
    addmm_77: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg266_1, view_324, permute_151);  arg266_1 = view_324 = permute_151 = None
    view_325: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_77, [8, 1, 196, 512]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_143: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_325);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_142: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_139, clone_143);  add_139 = clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_142, [3], correction = 0, keepdim = True)
    getitem_146: "f32[8, 1, 196, 1]" = var_mean_41[0]
    getitem_147: "f32[8, 1, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
    rsqrt_41: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_61: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_142, getitem_147);  getitem_147 = None
    mul_179: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_41);  sub_61 = rsqrt_41 = None
    mul_180: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_179, arg85_1);  mul_179 = arg85_1 = None
    add_144: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_180, arg86_1);  mul_180 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_326: "f32[1568, 512]" = torch.ops.aten.view.default(add_144, [1568, 512]);  add_144 = None
    permute_152: "f32[512, 2048]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    addmm_78: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg268_1, view_326, permute_152);  arg268_1 = view_326 = permute_152 = None
    view_327: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_78, [8, 1, 196, 2048]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_181: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
    mul_182: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
    erf_19: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
    add_145: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_183: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_181, add_145);  mul_181 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_144: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_183);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_328: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_144, [1568, 2048]);  clone_144 = None
    permute_153: "f32[2048, 512]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
    addmm_79: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg270_1, view_328, permute_153);  arg270_1 = view_328 = permute_153 = None
    view_329: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_79, [8, 1, 196, 512]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_145: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_329);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_146: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_142, clone_145);  add_142 = clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_146, [3], correction = 0, keepdim = True)
    getitem_148: "f32[8, 1, 196, 1]" = var_mean_42[0]
    getitem_149: "f32[8, 1, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_42: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_62: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_146, getitem_149);  getitem_149 = None
    mul_184: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = rsqrt_42 = None
    mul_185: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_184, arg87_1);  mul_184 = arg87_1 = None
    add_148: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_185, arg88_1);  mul_185 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_330: "f32[1568, 512]" = torch.ops.aten.view.default(add_148, [1568, 512]);  add_148 = None
    permute_154: "f32[512, 1536]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
    addmm_80: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg272_1, view_330, permute_154);  arg272_1 = view_330 = permute_154 = None
    view_331: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_80, [8, 1, 196, 1536]);  addmm_80 = None
    view_332: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_331, [8, 1, 196, 3, 16, 32]);  view_331 = None
    permute_155: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_332, [3, 0, 4, 1, 2, 5]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_155);  permute_155 = None
    getitem_150: "f32[8, 16, 1, 196, 32]" = unbind_20[0]
    getitem_151: "f32[8, 16, 1, 196, 32]" = unbind_20[1]
    getitem_152: "f32[8, 16, 1, 196, 32]" = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_186: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_150, 0.42044820762685725);  getitem_150 = None
    permute_156: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_151, [0, 1, 2, 4, 3]);  getitem_151 = None
    mul_187: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_156, 0.42044820762685725);  permute_156 = None
    expand_80: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_186, [8, 16, 1, 196, 32]);  mul_186 = None
    clone_146: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_333: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_146, [128, 196, 32]);  clone_146 = None
    expand_81: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_187, [8, 16, 1, 32, 196]);  mul_187 = None
    clone_147: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_334: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_147, [128, 32, 196]);  clone_147 = None
    bmm_40: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_333, view_334);  view_333 = view_334 = None
    view_335: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_40, [8, 16, 1, 196, 196]);  bmm_40 = None
    amax_20: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_63: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_335, amax_20);  view_335 = amax_20 = None
    exp_20: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_21: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    expand_82: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_20, [8, 16, 1, 196, 196]);  div_20 = None
    view_336: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_82, [128, 196, 196]);  expand_82 = None
    expand_83: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_152, [8, 16, 1, 196, 32]);  getitem_152 = None
    clone_148: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_337: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_148, [128, 196, 32]);  clone_148 = None
    bmm_41: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
    view_338: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_41, [8, 16, 1, 196, 32]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_157: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_338, [0, 2, 3, 4, 1]);  view_338 = None
    clone_149: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    view_339: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_149, [8, 1, 196, 512]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_340: "f32[1568, 512]" = torch.ops.aten.view.default(view_339, [1568, 512]);  view_339 = None
    permute_158: "f32[512, 512]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
    addmm_81: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg274_1, view_340, permute_158);  arg274_1 = view_340 = permute_158 = None
    view_341: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_81, [8, 1, 196, 512]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_150: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_341);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_149: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_146, clone_150);  add_146 = clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_149, [3], correction = 0, keepdim = True)
    getitem_153: "f32[8, 1, 196, 1]" = var_mean_43[0]
    getitem_154: "f32[8, 1, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-06);  getitem_153 = None
    rsqrt_43: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_64: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_149, getitem_154);  getitem_154 = None
    mul_188: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_43);  sub_64 = rsqrt_43 = None
    mul_189: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_188, arg89_1);  mul_188 = arg89_1 = None
    add_151: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_189, arg90_1);  mul_189 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_342: "f32[1568, 512]" = torch.ops.aten.view.default(add_151, [1568, 512]);  add_151 = None
    permute_159: "f32[512, 2048]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
    addmm_82: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg276_1, view_342, permute_159);  arg276_1 = view_342 = permute_159 = None
    view_343: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_82, [8, 1, 196, 2048]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_190: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    mul_191: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476);  view_343 = None
    erf_20: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
    add_152: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_192: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_190, add_152);  mul_190 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_151: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_192);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_344: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_151, [1568, 2048]);  clone_151 = None
    permute_160: "f32[2048, 512]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    addmm_83: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg278_1, view_344, permute_160);  arg278_1 = view_344 = permute_160 = None
    view_345: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_83, [8, 1, 196, 512]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_152: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_345);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_153: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_149, clone_152);  add_149 = clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_153, [3], correction = 0, keepdim = True)
    getitem_155: "f32[8, 1, 196, 1]" = var_mean_44[0]
    getitem_156: "f32[8, 1, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_155, 1e-06);  getitem_155 = None
    rsqrt_44: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_65: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_153, getitem_156);  getitem_156 = None
    mul_193: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = rsqrt_44 = None
    mul_194: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_193, arg91_1);  mul_193 = arg91_1 = None
    add_155: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_194, arg92_1);  mul_194 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_346: "f32[1568, 512]" = torch.ops.aten.view.default(add_155, [1568, 512]);  add_155 = None
    permute_161: "f32[512, 1536]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
    addmm_84: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg280_1, view_346, permute_161);  arg280_1 = view_346 = permute_161 = None
    view_347: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_84, [8, 1, 196, 1536]);  addmm_84 = None
    view_348: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_347, [8, 1, 196, 3, 16, 32]);  view_347 = None
    permute_162: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_348, [3, 0, 4, 1, 2, 5]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_162);  permute_162 = None
    getitem_157: "f32[8, 16, 1, 196, 32]" = unbind_21[0]
    getitem_158: "f32[8, 16, 1, 196, 32]" = unbind_21[1]
    getitem_159: "f32[8, 16, 1, 196, 32]" = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_195: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_157, 0.42044820762685725);  getitem_157 = None
    permute_163: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_158, [0, 1, 2, 4, 3]);  getitem_158 = None
    mul_196: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_163, 0.42044820762685725);  permute_163 = None
    expand_84: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_195, [8, 16, 1, 196, 32]);  mul_195 = None
    clone_153: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_349: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_153, [128, 196, 32]);  clone_153 = None
    expand_85: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_196, [8, 16, 1, 32, 196]);  mul_196 = None
    clone_154: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_350: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_154, [128, 32, 196]);  clone_154 = None
    bmm_42: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_349, view_350);  view_349 = view_350 = None
    view_351: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_42, [8, 16, 1, 196, 196]);  bmm_42 = None
    amax_21: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_351, [-1], True)
    sub_66: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_351, amax_21);  view_351 = amax_21 = None
    exp_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_22: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    expand_86: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_21, [8, 16, 1, 196, 196]);  div_21 = None
    view_352: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_86, [128, 196, 196]);  expand_86 = None
    expand_87: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_159, [8, 16, 1, 196, 32]);  getitem_159 = None
    clone_155: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    view_353: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_155, [128, 196, 32]);  clone_155 = None
    bmm_43: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_352, view_353);  view_352 = view_353 = None
    view_354: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_43, [8, 16, 1, 196, 32]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_164: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_354, [0, 2, 3, 4, 1]);  view_354 = None
    clone_156: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_355: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_156, [8, 1, 196, 512]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_356: "f32[1568, 512]" = torch.ops.aten.view.default(view_355, [1568, 512]);  view_355 = None
    permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
    addmm_85: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg282_1, view_356, permute_165);  arg282_1 = view_356 = permute_165 = None
    view_357: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_85, [8, 1, 196, 512]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_157: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_357);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_156: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_153, clone_157);  add_153 = clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_156, [3], correction = 0, keepdim = True)
    getitem_160: "f32[8, 1, 196, 1]" = var_mean_45[0]
    getitem_161: "f32[8, 1, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_45: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_67: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_156, getitem_161);  getitem_161 = None
    mul_197: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_45);  sub_67 = rsqrt_45 = None
    mul_198: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_197, arg93_1);  mul_197 = arg93_1 = None
    add_158: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_198, arg94_1);  mul_198 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_358: "f32[1568, 512]" = torch.ops.aten.view.default(add_158, [1568, 512]);  add_158 = None
    permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
    addmm_86: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg284_1, view_358, permute_166);  arg284_1 = view_358 = permute_166 = None
    view_359: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_86, [8, 1, 196, 2048]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_199: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.5)
    mul_200: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.7071067811865476);  view_359 = None
    erf_21: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_200);  mul_200 = None
    add_159: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_201: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_199, add_159);  mul_199 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_158: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_201);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_360: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_158, [1568, 2048]);  clone_158 = None
    permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
    addmm_87: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg286_1, view_360, permute_167);  arg286_1 = view_360 = permute_167 = None
    view_361: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_87, [8, 1, 196, 512]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_159: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_361);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_160: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_156, clone_159);  add_156 = clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_160, [3], correction = 0, keepdim = True)
    getitem_162: "f32[8, 1, 196, 1]" = var_mean_46[0]
    getitem_163: "f32[8, 1, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_46: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_68: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_160, getitem_163);  getitem_163 = None
    mul_202: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = rsqrt_46 = None
    mul_203: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_202, arg95_1);  mul_202 = arg95_1 = None
    add_162: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_203, arg96_1);  mul_203 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_362: "f32[1568, 512]" = torch.ops.aten.view.default(add_162, [1568, 512]);  add_162 = None
    permute_168: "f32[512, 1536]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
    addmm_88: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg288_1, view_362, permute_168);  arg288_1 = view_362 = permute_168 = None
    view_363: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_88, [8, 1, 196, 1536]);  addmm_88 = None
    view_364: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_363, [8, 1, 196, 3, 16, 32]);  view_363 = None
    permute_169: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_364, [3, 0, 4, 1, 2, 5]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_169);  permute_169 = None
    getitem_164: "f32[8, 16, 1, 196, 32]" = unbind_22[0]
    getitem_165: "f32[8, 16, 1, 196, 32]" = unbind_22[1]
    getitem_166: "f32[8, 16, 1, 196, 32]" = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_204: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_164, 0.42044820762685725);  getitem_164 = None
    permute_170: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_165, [0, 1, 2, 4, 3]);  getitem_165 = None
    mul_205: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_170, 0.42044820762685725);  permute_170 = None
    expand_88: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_204, [8, 16, 1, 196, 32]);  mul_204 = None
    clone_160: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_365: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_160, [128, 196, 32]);  clone_160 = None
    expand_89: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_205, [8, 16, 1, 32, 196]);  mul_205 = None
    clone_161: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_366: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_161, [128, 32, 196]);  clone_161 = None
    bmm_44: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_365, view_366);  view_365 = view_366 = None
    view_367: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_44, [8, 16, 1, 196, 196]);  bmm_44 = None
    amax_22: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_367, [-1], True)
    sub_69: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_367, amax_22);  view_367 = amax_22 = None
    exp_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_23: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    expand_90: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_22, [8, 16, 1, 196, 196]);  div_22 = None
    view_368: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_90, [128, 196, 196]);  expand_90 = None
    expand_91: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_166, [8, 16, 1, 196, 32]);  getitem_166 = None
    clone_162: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
    view_369: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_162, [128, 196, 32]);  clone_162 = None
    bmm_45: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_368, view_369);  view_368 = view_369 = None
    view_370: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_45, [8, 16, 1, 196, 32]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_171: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_370, [0, 2, 3, 4, 1]);  view_370 = None
    clone_163: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_371: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_163, [8, 1, 196, 512]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_372: "f32[1568, 512]" = torch.ops.aten.view.default(view_371, [1568, 512]);  view_371 = None
    permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
    addmm_89: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg290_1, view_372, permute_172);  arg290_1 = view_372 = permute_172 = None
    view_373: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_89, [8, 1, 196, 512]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_164: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_373);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_163: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_160, clone_164);  add_160 = clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_163, [3], correction = 0, keepdim = True)
    getitem_167: "f32[8, 1, 196, 1]" = var_mean_47[0]
    getitem_168: "f32[8, 1, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
    rsqrt_47: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_70: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_163, getitem_168);  getitem_168 = None
    mul_206: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_47);  sub_70 = rsqrt_47 = None
    mul_207: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_206, arg97_1);  mul_206 = arg97_1 = None
    add_165: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_207, arg98_1);  mul_207 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_374: "f32[1568, 512]" = torch.ops.aten.view.default(add_165, [1568, 512]);  add_165 = None
    permute_173: "f32[512, 2048]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
    addmm_90: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg292_1, view_374, permute_173);  arg292_1 = view_374 = permute_173 = None
    view_375: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_90, [8, 1, 196, 2048]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.5)
    mul_209: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.7071067811865476);  view_375 = None
    erf_22: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_166: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_210: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_208, add_166);  mul_208 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_165: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_210);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_376: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_165, [1568, 2048]);  clone_165 = None
    permute_174: "f32[2048, 512]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    addmm_91: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg294_1, view_376, permute_174);  arg294_1 = view_376 = permute_174 = None
    view_377: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_91, [8, 1, 196, 512]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_166: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_377);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_167: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_163, clone_166);  add_163 = clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_167, [3], correction = 0, keepdim = True)
    getitem_169: "f32[8, 1, 196, 1]" = var_mean_48[0]
    getitem_170: "f32[8, 1, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-06);  getitem_169 = None
    rsqrt_48: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_71: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_167, getitem_170);  getitem_170 = None
    mul_211: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_48);  sub_71 = rsqrt_48 = None
    mul_212: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_211, arg99_1);  mul_211 = arg99_1 = None
    add_169: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_212, arg100_1);  mul_212 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_378: "f32[1568, 512]" = torch.ops.aten.view.default(add_169, [1568, 512]);  add_169 = None
    permute_175: "f32[512, 1536]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
    addmm_92: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg296_1, view_378, permute_175);  arg296_1 = view_378 = permute_175 = None
    view_379: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_92, [8, 1, 196, 1536]);  addmm_92 = None
    view_380: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_379, [8, 1, 196, 3, 16, 32]);  view_379 = None
    permute_176: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_380, [3, 0, 4, 1, 2, 5]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_176);  permute_176 = None
    getitem_171: "f32[8, 16, 1, 196, 32]" = unbind_23[0]
    getitem_172: "f32[8, 16, 1, 196, 32]" = unbind_23[1]
    getitem_173: "f32[8, 16, 1, 196, 32]" = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_213: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_171, 0.42044820762685725);  getitem_171 = None
    permute_177: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_172, [0, 1, 2, 4, 3]);  getitem_172 = None
    mul_214: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_177, 0.42044820762685725);  permute_177 = None
    expand_92: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_213, [8, 16, 1, 196, 32]);  mul_213 = None
    clone_167: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_381: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_167, [128, 196, 32]);  clone_167 = None
    expand_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_214, [8, 16, 1, 32, 196]);  mul_214 = None
    clone_168: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_382: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_168, [128, 32, 196]);  clone_168 = None
    bmm_46: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_381, view_382);  view_381 = view_382 = None
    view_383: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_46, [8, 16, 1, 196, 196]);  bmm_46 = None
    amax_23: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_383, [-1], True)
    sub_72: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_383, amax_23);  view_383 = amax_23 = None
    exp_23: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
    sum_24: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    expand_94: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_23, [8, 16, 1, 196, 196]);  div_23 = None
    view_384: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_94, [128, 196, 196]);  expand_94 = None
    expand_95: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_173, [8, 16, 1, 196, 32]);  getitem_173 = None
    clone_169: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    view_385: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_169, [128, 196, 32]);  clone_169 = None
    bmm_47: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
    view_386: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_47, [8, 16, 1, 196, 32]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_178: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_386, [0, 2, 3, 4, 1]);  view_386 = None
    clone_170: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
    view_387: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_170, [8, 1, 196, 512]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_388: "f32[1568, 512]" = torch.ops.aten.view.default(view_387, [1568, 512]);  view_387 = None
    permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
    addmm_93: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg298_1, view_388, permute_179);  arg298_1 = view_388 = permute_179 = None
    view_389: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_93, [8, 1, 196, 512]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_171: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_389);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_170: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_167, clone_171);  add_167 = clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_170, [3], correction = 0, keepdim = True)
    getitem_174: "f32[8, 1, 196, 1]" = var_mean_49[0]
    getitem_175: "f32[8, 1, 196, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_49: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_73: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_170, getitem_175);  getitem_175 = None
    mul_215: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = rsqrt_49 = None
    mul_216: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_215, arg101_1);  mul_215 = arg101_1 = None
    add_172: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_216, arg102_1);  mul_216 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[1568, 512]" = torch.ops.aten.view.default(add_172, [1568, 512]);  add_172 = None
    permute_180: "f32[512, 2048]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
    addmm_94: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg300_1, view_390, permute_180);  arg300_1 = view_390 = permute_180 = None
    view_391: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_94, [8, 1, 196, 2048]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_217: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    mul_218: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476);  view_391 = None
    erf_23: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
    add_173: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_219: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_217, add_173);  mul_217 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_172: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_219);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_392: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_172, [1568, 2048]);  clone_172 = None
    permute_181: "f32[2048, 512]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
    addmm_95: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg302_1, view_392, permute_181);  arg302_1 = view_392 = permute_181 = None
    view_393: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_95, [8, 1, 196, 512]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_173: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_393);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_174: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_170, clone_173);  add_170 = clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_394: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.view.default(add_174, [8, 1, 1, 14, 14, 512]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_182: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.permute.default(view_394, [0, 1, 3, 2, 4, 5]);  view_394 = None
    view_395: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(permute_182, [8, 14, 14, 512]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_183: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_395, [0, 3, 1, 2]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_184: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(permute_183, [0, 2, 3, 1]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_50 = torch.ops.aten.var_mean.correction(permute_184, [3], correction = 0, keepdim = True)
    getitem_176: "f32[8, 14, 14, 1]" = var_mean_50[0]
    getitem_177: "f32[8, 14, 14, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_50: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_74: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_184, getitem_177);  permute_184 = getitem_177 = None
    mul_220: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_50);  sub_74 = rsqrt_50 = None
    mul_221: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_220, arg103_1);  mul_220 = arg103_1 = None
    add_176: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_221, arg104_1);  mul_221 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_185: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(add_176, [0, 3, 1, 2]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(permute_185, [-1, -2], True);  permute_185 = None
    as_strided: "f32[8, 512, 1, 1]" = torch.ops.aten.as_strided.default(mean, [8, 512, 1, 1], [512, 1, 512, 512]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_396: "f32[8, 512]" = torch.ops.aten.view.default(as_strided, [8, 512]);  as_strided = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:432, code: x = self.head_drop(x)
    clone_174: "f32[8, 512]" = torch.ops.aten.clone.default(view_396);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:433, code: return x if pre_logits else self.head(x)
    permute_186: "f32[512, 1000]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
    addmm_96: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg304_1, clone_174, permute_186);  arg304_1 = clone_174 = permute_186 = None
    return (addmm_96,)
    