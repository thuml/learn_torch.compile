from __future__ import annotations



def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[16, 1, 1, 1]", arg2_1: "f32[16]", arg3_1: "f32[32, 16, 3, 3]", arg4_1: "f32[32, 1, 1, 1]", arg5_1: "f32[32]", arg6_1: "f32[64, 32, 3, 3]", arg7_1: "f32[64, 1, 1, 1]", arg8_1: "f32[64]", arg9_1: "f32[128, 64, 3, 3]", arg10_1: "f32[128, 1, 1, 1]", arg11_1: "f32[128]", arg12_1: "f32[256, 128, 1, 1]", arg13_1: "f32[256, 1, 1, 1]", arg14_1: "f32[256]", arg15_1: "f32[64, 128, 1, 1]", arg16_1: "f32[64, 1, 1, 1]", arg17_1: "f32[64]", arg18_1: "f32[64, 64, 3, 3]", arg19_1: "f32[64, 1, 1, 1]", arg20_1: "f32[64]", arg21_1: "f32[64, 64, 3, 3]", arg22_1: "f32[64, 1, 1, 1]", arg23_1: "f32[64]", arg24_1: "f32[256, 64, 1, 1]", arg25_1: "f32[256, 1, 1, 1]", arg26_1: "f32[256]", arg27_1: "f32[512, 256, 1, 1]", arg28_1: "f32[512, 1, 1, 1]", arg29_1: "f32[512]", arg30_1: "f32[128, 256, 1, 1]", arg31_1: "f32[128, 1, 1, 1]", arg32_1: "f32[128]", arg33_1: "f32[128, 64, 3, 3]", arg34_1: "f32[128, 1, 1, 1]", arg35_1: "f32[128]", arg36_1: "f32[128, 64, 3, 3]", arg37_1: "f32[128, 1, 1, 1]", arg38_1: "f32[128]", arg39_1: "f32[512, 128, 1, 1]", arg40_1: "f32[512, 1, 1, 1]", arg41_1: "f32[512]", arg42_1: "f32[128, 512, 1, 1]", arg43_1: "f32[128, 1, 1, 1]", arg44_1: "f32[128]", arg45_1: "f32[128, 64, 3, 3]", arg46_1: "f32[128, 1, 1, 1]", arg47_1: "f32[128]", arg48_1: "f32[128, 64, 3, 3]", arg49_1: "f32[128, 1, 1, 1]", arg50_1: "f32[128]", arg51_1: "f32[512, 128, 1, 1]", arg52_1: "f32[512, 1, 1, 1]", arg53_1: "f32[512]", arg54_1: "f32[1536, 512, 1, 1]", arg55_1: "f32[1536, 1, 1, 1]", arg56_1: "f32[1536]", arg57_1: "f32[384, 512, 1, 1]", arg58_1: "f32[384, 1, 1, 1]", arg59_1: "f32[384]", arg60_1: "f32[384, 64, 3, 3]", arg61_1: "f32[384, 1, 1, 1]", arg62_1: "f32[384]", arg63_1: "f32[384, 64, 3, 3]", arg64_1: "f32[384, 1, 1, 1]", arg65_1: "f32[384]", arg66_1: "f32[1536, 384, 1, 1]", arg67_1: "f32[1536, 1, 1, 1]", arg68_1: "f32[1536]", arg69_1: "f32[384, 1536, 1, 1]", arg70_1: "f32[384, 1, 1, 1]", arg71_1: "f32[384]", arg72_1: "f32[384, 64, 3, 3]", arg73_1: "f32[384, 1, 1, 1]", arg74_1: "f32[384]", arg75_1: "f32[384, 64, 3, 3]", arg76_1: "f32[384, 1, 1, 1]", arg77_1: "f32[384]", arg78_1: "f32[1536, 384, 1, 1]", arg79_1: "f32[1536, 1, 1, 1]", arg80_1: "f32[1536]", arg81_1: "f32[384, 1536, 1, 1]", arg82_1: "f32[384, 1, 1, 1]", arg83_1: "f32[384]", arg84_1: "f32[384, 64, 3, 3]", arg85_1: "f32[384, 1, 1, 1]", arg86_1: "f32[384]", arg87_1: "f32[384, 64, 3, 3]", arg88_1: "f32[384, 1, 1, 1]", arg89_1: "f32[384]", arg90_1: "f32[1536, 384, 1, 1]", arg91_1: "f32[1536, 1, 1, 1]", arg92_1: "f32[1536]", arg93_1: "f32[384, 1536, 1, 1]", arg94_1: "f32[384, 1, 1, 1]", arg95_1: "f32[384]", arg96_1: "f32[384, 64, 3, 3]", arg97_1: "f32[384, 1, 1, 1]", arg98_1: "f32[384]", arg99_1: "f32[384, 64, 3, 3]", arg100_1: "f32[384, 1, 1, 1]", arg101_1: "f32[384]", arg102_1: "f32[1536, 384, 1, 1]", arg103_1: "f32[1536, 1, 1, 1]", arg104_1: "f32[1536]", arg105_1: "f32[384, 1536, 1, 1]", arg106_1: "f32[384, 1, 1, 1]", arg107_1: "f32[384]", arg108_1: "f32[384, 64, 3, 3]", arg109_1: "f32[384, 1, 1, 1]", arg110_1: "f32[384]", arg111_1: "f32[384, 64, 3, 3]", arg112_1: "f32[384, 1, 1, 1]", arg113_1: "f32[384]", arg114_1: "f32[1536, 384, 1, 1]", arg115_1: "f32[1536, 1, 1, 1]", arg116_1: "f32[1536]", arg117_1: "f32[384, 1536, 1, 1]", arg118_1: "f32[384, 1, 1, 1]", arg119_1: "f32[384]", arg120_1: "f32[384, 64, 3, 3]", arg121_1: "f32[384, 1, 1, 1]", arg122_1: "f32[384]", arg123_1: "f32[384, 64, 3, 3]", arg124_1: "f32[384, 1, 1, 1]", arg125_1: "f32[384]", arg126_1: "f32[1536, 384, 1, 1]", arg127_1: "f32[1536, 1, 1, 1]", arg128_1: "f32[1536]", arg129_1: "f32[1536, 1536, 1, 1]", arg130_1: "f32[1536, 1, 1, 1]", arg131_1: "f32[1536]", arg132_1: "f32[384, 1536, 1, 1]", arg133_1: "f32[384, 1, 1, 1]", arg134_1: "f32[384]", arg135_1: "f32[384, 64, 3, 3]", arg136_1: "f32[384, 1, 1, 1]", arg137_1: "f32[384]", arg138_1: "f32[384, 64, 3, 3]", arg139_1: "f32[384, 1, 1, 1]", arg140_1: "f32[384]", arg141_1: "f32[1536, 384, 1, 1]", arg142_1: "f32[1536, 1, 1, 1]", arg143_1: "f32[1536]", arg144_1: "f32[384, 1536, 1, 1]", arg145_1: "f32[384, 1, 1, 1]", arg146_1: "f32[384]", arg147_1: "f32[384, 64, 3, 3]", arg148_1: "f32[384, 1, 1, 1]", arg149_1: "f32[384]", arg150_1: "f32[384, 64, 3, 3]", arg151_1: "f32[384, 1, 1, 1]", arg152_1: "f32[384]", arg153_1: "f32[1536, 384, 1, 1]", arg154_1: "f32[1536, 1, 1, 1]", arg155_1: "f32[1536]", arg156_1: "f32[384, 1536, 1, 1]", arg157_1: "f32[384, 1, 1, 1]", arg158_1: "f32[384]", arg159_1: "f32[384, 64, 3, 3]", arg160_1: "f32[384, 1, 1, 1]", arg161_1: "f32[384]", arg162_1: "f32[384, 64, 3, 3]", arg163_1: "f32[384, 1, 1, 1]", arg164_1: "f32[384]", arg165_1: "f32[1536, 384, 1, 1]", arg166_1: "f32[1536, 1, 1, 1]", arg167_1: "f32[1536]", arg168_1: "f32[2304, 1536, 1, 1]", arg169_1: "f32[2304, 1, 1, 1]", arg170_1: "f32[2304]", arg171_1: "f32[64, 256, 1, 1]", arg172_1: "f32[64]", arg173_1: "f32[256, 64, 1, 1]", arg174_1: "f32[256]", arg175_1: "f32[128, 512, 1, 1]", arg176_1: "f32[128]", arg177_1: "f32[512, 128, 1, 1]", arg178_1: "f32[512]", arg179_1: "f32[128, 512, 1, 1]", arg180_1: "f32[128]", arg181_1: "f32[512, 128, 1, 1]", arg182_1: "f32[512]", arg183_1: "f32[384, 1536, 1, 1]", arg184_1: "f32[384]", arg185_1: "f32[1536, 384, 1, 1]", arg186_1: "f32[1536]", arg187_1: "f32[384, 1536, 1, 1]", arg188_1: "f32[384]", arg189_1: "f32[1536, 384, 1, 1]", arg190_1: "f32[1536]", arg191_1: "f32[384, 1536, 1, 1]", arg192_1: "f32[384]", arg193_1: "f32[1536, 384, 1, 1]", arg194_1: "f32[1536]", arg195_1: "f32[384, 1536, 1, 1]", arg196_1: "f32[384]", arg197_1: "f32[1536, 384, 1, 1]", arg198_1: "f32[1536]", arg199_1: "f32[384, 1536, 1, 1]", arg200_1: "f32[384]", arg201_1: "f32[1536, 384, 1, 1]", arg202_1: "f32[1536]", arg203_1: "f32[384, 1536, 1, 1]", arg204_1: "f32[384]", arg205_1: "f32[1536, 384, 1, 1]", arg206_1: "f32[1536]", arg207_1: "f32[384, 1536, 1, 1]", arg208_1: "f32[384]", arg209_1: "f32[1536, 384, 1, 1]", arg210_1: "f32[1536]", arg211_1: "f32[384, 1536, 1, 1]", arg212_1: "f32[384]", arg213_1: "f32[1536, 384, 1, 1]", arg214_1: "f32[1536]", arg215_1: "f32[384, 1536, 1, 1]", arg216_1: "f32[384]", arg217_1: "f32[1536, 384, 1, 1]", arg218_1: "f32[1536]", arg219_1: "f32[1000, 2304]", arg220_1: "f32[1000]", arg221_1: "f32[8, 3, 288, 288]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view: "f32[1, 16, 27]" = torch.ops.aten.view.default(arg0_1, [1, 16, -1]);  arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg1_1, 0.34412564994580647);  arg1_1 = None
    view_1: "f32[16]" = torch.ops.aten.view.default(mul, [-1]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(view, [0, 2], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
    mul_1: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
    mul_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze);  mul_1 = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_2: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_2, [16, 3, 3, 3]);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution: "f32[8, 16, 144, 144]" = torch.ops.aten.convolution.default(arg221_1, view_2, arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg221_1 = view_2 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid: "f32[8, 16, 144, 144]" = torch.ops.aten.sigmoid.default(convolution)
    mul_3: "f32[8, 16, 144, 144]" = torch.ops.aten.mul.Tensor(convolution, sigmoid);  convolution = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_3: "f32[1, 32, 144]" = torch.ops.aten.view.default(arg3_1, [1, 32, -1]);  arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_4: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg4_1, 0.1490107774734497);  arg4_1 = None
    view_4: "f32[32]" = torch.ops.aten.view.default(mul_4, [-1]);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [0, 2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1]" = var_mean_1[1];  var_mean_1 = None
    add_1: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub_1: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = getitem_3 = None
    mul_5: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    unsqueeze_1: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_4, -1);  view_4 = None
    mul_6: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_1);  mul_5 = unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_5: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_6, [32, 16, 3, 3]);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_1: "f32[8, 32, 144, 144]" = torch.ops.aten.convolution.default(mul_3, view_5, arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_3 = view_5 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_1: "f32[8, 32, 144, 144]" = torch.ops.aten.sigmoid.default(convolution_1)
    mul_7: "f32[8, 32, 144, 144]" = torch.ops.aten.mul.Tensor(convolution_1, sigmoid_1);  convolution_1 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_6: "f32[1, 64, 288]" = torch.ops.aten.view.default(arg6_1, [1, 64, -1]);  arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_8: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg7_1, 0.10536653122135592);  arg7_1 = None
    view_7: "f32[64]" = torch.ops.aten.view.default(mul_8, [-1]);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [0, 2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None
    add_2: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_2: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = getitem_5 = None
    mul_9: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_7, -1);  view_7 = None
    mul_10: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_2);  mul_9 = unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_8: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_10, [64, 32, 3, 3]);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_2: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(mul_7, view_8, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_7 = view_8 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_2: "f32[8, 64, 144, 144]" = torch.ops.aten.sigmoid.default(convolution_2)
    mul_11: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(convolution_2, sigmoid_2);  convolution_2 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_9: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg9_1, [1, 128, -1]);  arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_12: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg10_1, 0.07450538873672485);  arg10_1 = None
    view_10: "f32[128]" = torch.ops.aten.view.default(mul_12, [-1]);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [0, 2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_3: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, getitem_7);  view_9 = getitem_7 = None
    mul_13: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    unsqueeze_3: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_10, -1);  view_10 = None
    mul_14: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_3);  mul_13 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_11: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_14, [128, 64, 3, 3]);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_3: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(mul_11, view_11, arg11_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_11 = view_11 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_3: "f32[8, 128, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_3)
    mul_15: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_3, sigmoid_3);  convolution_3 = sigmoid_3 = None
    mul_16: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_15, 1.0);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_12: "f32[1, 256, 128]" = torch.ops.aten.view.default(arg12_1, [1, 256, -1]);  arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_17: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg13_1, 0.1580497968320339);  arg13_1 = None
    view_13: "f32[256]" = torch.ops.aten.view.default(mul_17, [-1]);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [0, 2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 256, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 256, 1]" = var_mean_4[1];  var_mean_4 = None
    add_4: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_4: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = getitem_9 = None
    mul_18: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    unsqueeze_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_13, -1);  view_13 = None
    mul_19: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_4);  mul_18 = unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_14: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_19, [256, 128, 1, 1]);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_4: "f32[8, 256, 72, 72]" = torch.ops.aten.convolution.default(mul_16, view_14, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_14 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_15: "f32[1, 64, 128]" = torch.ops.aten.view.default(arg15_1, [1, 64, -1]);  arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_20: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg16_1, 0.1580497968320339);  arg16_1 = None
    view_16: "f32[64]" = torch.ops.aten.view.default(mul_20, [-1]);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0, 2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 64, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 64, 1]" = var_mean_5[1];  var_mean_5 = None
    add_5: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_5: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = getitem_11 = None
    mul_21: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    unsqueeze_5: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_16, -1);  view_16 = None
    mul_22: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_5);  mul_21 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_17: "f32[64, 128, 1, 1]" = torch.ops.aten.view.default(mul_22, [64, 128, 1, 1]);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_5: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(mul_16, view_17, arg17_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_16 = view_17 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_4: "f32[8, 64, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_5)
    mul_23: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_5, sigmoid_4);  convolution_5 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_18: "f32[1, 64, 576]" = torch.ops.aten.view.default(arg18_1, [1, 64, -1]);  arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_24: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg19_1, 0.07450538873672485);  arg19_1 = None
    view_19: "f32[64]" = torch.ops.aten.view.default(mul_24, [-1]);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [0, 2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 64, 1]" = var_mean_6[1];  var_mean_6 = None
    add_6: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_6: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_18, getitem_13);  view_18 = getitem_13 = None
    mul_25: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_19, -1);  view_19 = None
    mul_26: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_6);  mul_25 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_20: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_26, [64, 64, 3, 3]);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_6: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(mul_23, view_20, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_23 = view_20 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_5: "f32[8, 64, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_6)
    mul_27: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_6, sigmoid_5);  convolution_6 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_21: "f32[1, 64, 576]" = torch.ops.aten.view.default(arg21_1, [1, 64, -1]);  arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_28: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg22_1, 0.07450538873672485);  arg22_1 = None
    view_22: "f32[64]" = torch.ops.aten.view.default(mul_28, [-1]);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [0, 2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 64, 1]" = var_mean_7[1];  var_mean_7 = None
    add_7: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_7: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_21, getitem_15);  view_21 = getitem_15 = None
    mul_29: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    unsqueeze_7: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_22, -1);  view_22 = None
    mul_30: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_7);  mul_29 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_23: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_30, [64, 64, 3, 3]);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_7: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(mul_27, view_23, arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_27 = view_23 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_6: "f32[8, 64, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_7)
    mul_31: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_7, sigmoid_6);  convolution_7 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_24: "f32[1, 256, 64]" = torch.ops.aten.view.default(arg24_1, [1, 256, -1]);  arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_32: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg25_1, 0.22351616621017456);  arg25_1 = None
    view_25: "f32[256]" = torch.ops.aten.view.default(mul_32, [-1]);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [0, 2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 256, 1]" = var_mean_8[1];  var_mean_8 = None
    add_8: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_8: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_24, getitem_17);  view_24 = getitem_17 = None
    mul_33: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    unsqueeze_8: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_25, -1);  view_25 = None
    mul_34: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_8);  mul_33 = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_26: "f32[256, 64, 1, 1]" = torch.ops.aten.view.default(mul_34, [256, 64, 1, 1]);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_8: "f32[8, 256, 72, 72]" = torch.ops.aten.convolution.default(mul_31, view_26, arg26_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_31 = view_26 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_8, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_9: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean, arg171_1, arg172_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg171_1 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_10: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu, arg173_1, arg174_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg173_1 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_35: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid_7);  convolution_8 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_36: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_35, 2.0);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_37: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_36, 0.2);  mul_36 = None
    add_9: "f32[8, 256, 72, 72]" = torch.ops.aten.add.Tensor(mul_37, convolution_4);  mul_37 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_8: "f32[8, 256, 72, 72]" = torch.ops.aten.sigmoid.default(add_9)
    mul_38: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_8);  add_9 = sigmoid_8 = None
    mul_39: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_38, 0.9805806756909201);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d: "f32[8, 256, 36, 36]" = torch.ops.aten.avg_pool2d.default(mul_39, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(arg27_1, [1, 512, -1]);  arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_40: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg28_1, 0.11175808310508728);  arg28_1 = None
    view_28: "f32[512]" = torch.ops.aten.view.default(mul_40, [-1]);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [0, 2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_19);  view_27 = getitem_19 = None
    mul_41: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    unsqueeze_9: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_28, -1);  view_28 = None
    mul_42: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_9);  mul_41 = unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_29: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_42, [512, 256, 1, 1]);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_11: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(avg_pool2d, view_29, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d = view_29 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_30: "f32[1, 128, 256]" = torch.ops.aten.view.default(arg30_1, [1, 128, -1]);  arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_43: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg31_1, 0.11175808310508728);  arg31_1 = None
    view_31: "f32[128]" = torch.ops.aten.view.default(mul_43, [-1]);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [0, 2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_10: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = getitem_21 = None
    mul_44: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_31, -1);  view_31 = None
    mul_45: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_10);  mul_44 = unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_32: "f32[128, 256, 1, 1]" = torch.ops.aten.view.default(mul_45, [128, 256, 1, 1]);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_12: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(mul_39, view_32, arg32_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_39 = view_32 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_9: "f32[8, 128, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_12)
    mul_46: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_12, sigmoid_9);  convolution_12 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_33: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg33_1, [1, 128, -1]);  arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_47: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg34_1, 0.07450538873672485);  arg34_1 = None
    view_34: "f32[128]" = torch.ops.aten.view.default(mul_47, [-1]);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [0, 2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_11: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_33, getitem_23);  view_33 = getitem_23 = None
    mul_48: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    unsqueeze_11: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_34, -1);  view_34 = None
    mul_49: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_11);  mul_48 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_35: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_49, [128, 64, 3, 3]);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_13: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_46, view_35, arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2);  mul_46 = view_35 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_10: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_13)
    mul_50: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_13, sigmoid_10);  convolution_13 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_36: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg36_1, [1, 128, -1]);  arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_51: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg37_1, 0.07450538873672485);  arg37_1 = None
    view_37: "f32[128]" = torch.ops.aten.view.default(mul_51, [-1]);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [0, 2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_13: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_12: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_36, getitem_25);  view_36 = getitem_25 = None
    mul_52: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_37, -1);  view_37 = None
    mul_53: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_12);  mul_52 = unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_38: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_53, [128, 64, 3, 3]);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_14: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_50, view_38, arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_50 = view_38 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_11: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_14)
    mul_54: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_14, sigmoid_11);  convolution_14 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_39: "f32[1, 512, 128]" = torch.ops.aten.view.default(arg39_1, [1, 512, -1]);  arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_55: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg40_1, 0.1580497968320339);  arg40_1 = None
    view_40: "f32[512]" = torch.ops.aten.view.default(mul_55, [-1]);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [0, 2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_13: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_39, getitem_27);  view_39 = getitem_27 = None
    mul_56: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    unsqueeze_13: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_40, -1);  view_40 = None
    mul_57: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_13);  mul_56 = unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_41: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_57, [512, 128, 1, 1]);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_15: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(mul_54, view_41, arg41_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_54 = view_41 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_15, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_16: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg175_1, arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg175_1 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_1: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_17: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_1, arg177_1, arg178_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg177_1 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_58: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_12);  convolution_15 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_59: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_58, 2.0);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_60: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_59, 0.2);  mul_59 = None
    add_15: "f32[8, 512, 36, 36]" = torch.ops.aten.add.Tensor(mul_60, convolution_11);  mul_60 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_13: "f32[8, 512, 36, 36]" = torch.ops.aten.sigmoid.default(add_15)
    mul_61: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(add_15, sigmoid_13);  sigmoid_13 = None
    mul_62: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_61, 0.9805806756909201);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_42: "f32[1, 128, 512]" = torch.ops.aten.view.default(arg42_1, [1, 128, -1]);  arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_63: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg43_1, 0.07902489841601695);  arg43_1 = None
    view_43: "f32[128]" = torch.ops.aten.view.default(mul_63, [-1]);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [0, 2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_16: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_14: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_42, getitem_29);  view_42 = getitem_29 = None
    mul_64: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_43, -1);  view_43 = None
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_14);  mul_64 = unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_44: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(mul_65, [128, 512, 1, 1]);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_18: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_62, view_44, arg44_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_62 = view_44 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_14: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_18)
    mul_66: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_18, sigmoid_14);  convolution_18 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_45: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg45_1, [1, 128, -1]);  arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_67: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg46_1, 0.07450538873672485);  arg46_1 = None
    view_46: "f32[128]" = torch.ops.aten.view.default(mul_67, [-1]);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [0, 2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_15: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_45, getitem_31);  view_45 = getitem_31 = None
    mul_68: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    unsqueeze_15: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_46, -1);  view_46 = None
    mul_69: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_15);  mul_68 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_47: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_69, [128, 64, 3, 3]);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_19: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_66, view_47, arg47_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_66 = view_47 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_15: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_19)
    mul_70: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_19, sigmoid_15);  convolution_19 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_48: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg48_1, [1, 128, -1]);  arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_71: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg49_1, 0.07450538873672485);  arg49_1 = None
    view_49: "f32[128]" = torch.ops.aten.view.default(mul_71, [-1]);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [0, 2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_16: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_48, getitem_33);  view_48 = getitem_33 = None
    mul_72: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_49, -1);  view_49 = None
    mul_73: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_16);  mul_72 = unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_50: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_73, [128, 64, 3, 3]);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_20: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_70, view_50, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_70 = view_50 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_16: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_20)
    mul_74: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_20, sigmoid_16);  convolution_20 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_51: "f32[1, 512, 128]" = torch.ops.aten.view.default(arg51_1, [1, 512, -1]);  arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_75: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg52_1, 0.1580497968320339);  arg52_1 = None
    view_52: "f32[512]" = torch.ops.aten.view.default(mul_75, [-1]);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [0, 2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_19: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_17: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_51, getitem_35);  view_51 = getitem_35 = None
    mul_76: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    unsqueeze_17: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_52, -1);  view_52 = None
    mul_77: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_17);  mul_76 = unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_53: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_77, [512, 128, 1, 1]);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_21: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(mul_74, view_53, arg53_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_74 = view_53 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_21, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_22: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg179_1, arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg179_1 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_2: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_23: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_2, arg181_1, arg182_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg181_1 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_78: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_17);  convolution_21 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_79: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_78, 2.0);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_80: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_79, 0.2);  mul_79 = None
    add_20: "f32[8, 512, 36, 36]" = torch.ops.aten.add.Tensor(mul_80, add_15);  mul_80 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_18: "f32[8, 512, 36, 36]" = torch.ops.aten.sigmoid.default(add_20)
    mul_81: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(add_20, sigmoid_18);  add_20 = sigmoid_18 = None
    mul_82: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_81, 0.9622504486493761);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_1: "f32[8, 512, 18, 18]" = torch.ops.aten.avg_pool2d.default(mul_82, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_54: "f32[1, 1536, 512]" = torch.ops.aten.view.default(arg54_1, [1, 1536, -1]);  arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_83: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg55_1, 0.07902489841601695);  arg55_1 = None
    view_55: "f32[1536]" = torch.ops.aten.view.default(mul_83, [-1]);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [0, 2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1536, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1536, 1]" = var_mean_18[1];  var_mean_18 = None
    add_21: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_18: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, getitem_37);  view_54 = getitem_37 = None
    mul_84: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    unsqueeze_18: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_55, -1);  view_55 = None
    mul_85: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_18);  mul_84 = unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_56: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_85, [1536, 512, 1, 1]);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_24: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(avg_pool2d_1, view_56, arg56_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_1 = view_56 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_57: "f32[1, 384, 512]" = torch.ops.aten.view.default(arg57_1, [1, 384, -1]);  arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_86: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg58_1, 0.07902489841601695);  arg58_1 = None
    view_58: "f32[384]" = torch.ops.aten.view.default(mul_86, [-1]);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [0, 2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 384, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 384, 1]" = var_mean_19[1];  var_mean_19 = None
    add_22: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_19: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_57, getitem_39);  view_57 = getitem_39 = None
    mul_87: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    unsqueeze_19: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_58, -1);  view_58 = None
    mul_88: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_19);  mul_87 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_59: "f32[384, 512, 1, 1]" = torch.ops.aten.view.default(mul_88, [384, 512, 1, 1]);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_25: "f32[8, 384, 36, 36]" = torch.ops.aten.convolution.default(mul_82, view_59, arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_82 = view_59 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_19: "f32[8, 384, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_25)
    mul_89: "f32[8, 384, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_25, sigmoid_19);  convolution_25 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_60: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg60_1, [1, 384, -1]);  arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_90: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg61_1, 0.07450538873672485);  arg61_1 = None
    view_61: "f32[384]" = torch.ops.aten.view.default(mul_90, [-1]);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [0, 2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 384, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 384, 1]" = var_mean_20[1];  var_mean_20 = None
    add_23: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_20: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_60, getitem_41);  view_60 = getitem_41 = None
    mul_91: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    unsqueeze_20: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_61, -1);  view_61 = None
    mul_92: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_20);  mul_91 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_62: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_92, [384, 64, 3, 3]);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_26: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_89, view_62, arg62_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  mul_89 = view_62 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_20: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_26)
    mul_93: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_26, sigmoid_20);  convolution_26 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_63: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg63_1, [1, 384, -1]);  arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_94: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg64_1, 0.07450538873672485);  arg64_1 = None
    view_64: "f32[384]" = torch.ops.aten.view.default(mul_94, [-1]);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [0, 2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 384, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 384, 1]" = var_mean_21[1];  var_mean_21 = None
    add_24: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_21: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_63, getitem_43);  view_63 = getitem_43 = None
    mul_95: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    unsqueeze_21: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_64, -1);  view_64 = None
    mul_96: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_21);  mul_95 = unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_65: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_96, [384, 64, 3, 3]);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_27: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_93, view_65, arg65_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_93 = view_65 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_21: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_97: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_21);  convolution_27 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_66: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg66_1, [1, 1536, -1]);  arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_98: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg67_1, 0.09125009274634042);  arg67_1 = None
    view_67: "f32[1536]" = torch.ops.aten.view.default(mul_98, [-1]);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [0, 2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1536, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1536, 1]" = var_mean_22[1];  var_mean_22 = None
    add_25: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_22: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_66, getitem_45);  view_66 = getitem_45 = None
    mul_99: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    unsqueeze_22: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_67, -1);  view_67 = None
    mul_100: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_22);  mul_99 = unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_68: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_100, [1536, 384, 1, 1]);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_28: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_97, view_68, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_97 = view_68 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_28, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_29: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg183_1, arg184_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg183_1 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_30: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_3, arg185_1, arg186_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_3 = arg185_1 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_22: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_101: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_22);  convolution_28 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_102: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_101, 2.0);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_103: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_102, 0.2);  mul_102 = None
    add_26: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_103, convolution_24);  mul_103 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_23: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_26)
    mul_104: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_26, sigmoid_23);  sigmoid_23 = None
    mul_105: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_104, 0.9805806756909201);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_69: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg69_1, [1, 384, -1]);  arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_106: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg70_1, 0.04562504637317021);  arg70_1 = None
    view_70: "f32[384]" = torch.ops.aten.view.default(mul_106, [-1]);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [0, 2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 384, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 384, 1]" = var_mean_23[1];  var_mean_23 = None
    add_27: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_23: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_69, getitem_47);  view_69 = getitem_47 = None
    mul_107: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    unsqueeze_23: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_70, -1);  view_70 = None
    mul_108: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_23);  mul_107 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_71: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_108, [384, 1536, 1, 1]);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_31: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_105, view_71, arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_105 = view_71 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_24: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_31)
    mul_109: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_31, sigmoid_24);  convolution_31 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_72: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg72_1, [1, 384, -1]);  arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_110: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg73_1, 0.07450538873672485);  arg73_1 = None
    view_73: "f32[384]" = torch.ops.aten.view.default(mul_110, [-1]);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [0, 2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 384, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 384, 1]" = var_mean_24[1];  var_mean_24 = None
    add_28: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_24: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_72, getitem_49);  view_72 = getitem_49 = None
    mul_111: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    unsqueeze_24: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_73, -1);  view_73 = None
    mul_112: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_24);  mul_111 = unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_74: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_112, [384, 64, 3, 3]);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_32: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_109, view_74, arg74_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_109 = view_74 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_25: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_32)
    mul_113: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_32, sigmoid_25);  convolution_32 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_75: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg75_1, [1, 384, -1]);  arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_114: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg76_1, 0.07450538873672485);  arg76_1 = None
    view_76: "f32[384]" = torch.ops.aten.view.default(mul_114, [-1]);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [0, 2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 384, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 384, 1]" = var_mean_25[1];  var_mean_25 = None
    add_29: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_25: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_75, getitem_51);  view_75 = getitem_51 = None
    mul_115: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
    unsqueeze_25: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_76, -1);  view_76 = None
    mul_116: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_25);  mul_115 = unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_77: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_116, [384, 64, 3, 3]);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_33: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_113, view_77, arg77_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_113 = view_77 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_26: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_33)
    mul_117: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_33, sigmoid_26);  convolution_33 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_78: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg78_1, [1, 1536, -1]);  arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_118: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg79_1, 0.09125009274634042);  arg79_1 = None
    view_79: "f32[1536]" = torch.ops.aten.view.default(mul_118, [-1]);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [0, 2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 1536, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 1536, 1]" = var_mean_26[1];  var_mean_26 = None
    add_30: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_26: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_78, getitem_53);  view_78 = getitem_53 = None
    mul_119: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
    unsqueeze_26: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_79, -1);  view_79 = None
    mul_120: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_26);  mul_119 = unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_80: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_120, [1536, 384, 1, 1]);  mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_34: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_117, view_80, arg80_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_117 = view_80 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_35: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_4, arg187_1, arg188_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg187_1 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_4: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_36: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_4, arg189_1, arg190_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg189_1 = arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_27: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_121: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_27);  convolution_34 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_122: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_121, 2.0);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_123: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_122, 0.2);  mul_122 = None
    add_31: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_123, add_26);  mul_123 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_28: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_31)
    mul_124: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_31, sigmoid_28);  sigmoid_28 = None
    mul_125: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_124, 0.9622504486493761);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_81: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg81_1, [1, 384, -1]);  arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_126: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg82_1, 0.04562504637317021);  arg82_1 = None
    view_82: "f32[384]" = torch.ops.aten.view.default(mul_126, [-1]);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [0, 2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 384, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 384, 1]" = var_mean_27[1];  var_mean_27 = None
    add_32: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_27: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_81, getitem_55);  view_81 = getitem_55 = None
    mul_127: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
    unsqueeze_27: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_82, -1);  view_82 = None
    mul_128: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_27);  mul_127 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_83: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_128, [384, 1536, 1, 1]);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_37: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_125, view_83, arg83_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_125 = view_83 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_29: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_37)
    mul_129: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_37, sigmoid_29);  convolution_37 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_84: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg84_1, [1, 384, -1]);  arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_130: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg85_1, 0.07450538873672485);  arg85_1 = None
    view_85: "f32[384]" = torch.ops.aten.view.default(mul_130, [-1]);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [0, 2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 384, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 384, 1]" = var_mean_28[1];  var_mean_28 = None
    add_33: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_28: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_84, getitem_57);  view_84 = getitem_57 = None
    mul_131: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
    unsqueeze_28: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_85, -1);  view_85 = None
    mul_132: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_28);  mul_131 = unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_86: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_132, [384, 64, 3, 3]);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_38: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_129, view_86, arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_129 = view_86 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_30: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_38)
    mul_133: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_38, sigmoid_30);  convolution_38 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_87: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg87_1, [1, 384, -1]);  arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_134: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg88_1, 0.07450538873672485);  arg88_1 = None
    view_88: "f32[384]" = torch.ops.aten.view.default(mul_134, [-1]);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [0, 2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 384, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 384, 1]" = var_mean_29[1];  var_mean_29 = None
    add_34: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_29: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_87, getitem_59);  view_87 = getitem_59 = None
    mul_135: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
    unsqueeze_29: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_88, -1);  view_88 = None
    mul_136: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_29);  mul_135 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_89: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_136, [384, 64, 3, 3]);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_39: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_133, view_89, arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_133 = view_89 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_31: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_39)
    mul_137: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_39, sigmoid_31);  convolution_39 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_90: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg90_1, [1, 1536, -1]);  arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_138: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg91_1, 0.09125009274634042);  arg91_1 = None
    view_91: "f32[1536]" = torch.ops.aten.view.default(mul_138, [-1]);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [0, 2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 1536, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 1536, 1]" = var_mean_30[1];  var_mean_30 = None
    add_35: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_30: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_90, getitem_61);  view_90 = getitem_61 = None
    mul_139: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
    unsqueeze_30: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_91, -1);  view_91 = None
    mul_140: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_30);  mul_139 = unsqueeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_92: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_140, [1536, 384, 1, 1]);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_40: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_137, view_92, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_137 = view_92 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_40, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_41: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg191_1, arg192_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg191_1 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_5: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_42: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_5, arg193_1, arg194_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg193_1 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_32: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_141: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_32);  convolution_40 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_142: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_141, 2.0);  mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_143: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_142, 0.2);  mul_142 = None
    add_36: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_143, add_31);  mul_143 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_33: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_36)
    mul_144: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_36, sigmoid_33);  sigmoid_33 = None
    mul_145: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_144, 0.9449111825230679);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_93: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg93_1, [1, 384, -1]);  arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_146: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg94_1, 0.04562504637317021);  arg94_1 = None
    view_94: "f32[384]" = torch.ops.aten.view.default(mul_146, [-1]);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [0, 2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 384, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 384, 1]" = var_mean_31[1];  var_mean_31 = None
    add_37: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_31: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_93, getitem_63);  view_93 = getitem_63 = None
    mul_147: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
    unsqueeze_31: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_94, -1);  view_94 = None
    mul_148: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_31);  mul_147 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_95: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_148, [384, 1536, 1, 1]);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_43: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_145, view_95, arg95_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_145 = view_95 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_34: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_149: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_43, sigmoid_34);  convolution_43 = sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_96: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg96_1, [1, 384, -1]);  arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_150: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg97_1, 0.07450538873672485);  arg97_1 = None
    view_97: "f32[384]" = torch.ops.aten.view.default(mul_150, [-1]);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [0, 2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 384, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 384, 1]" = var_mean_32[1];  var_mean_32 = None
    add_38: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_32: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_96, getitem_65);  view_96 = getitem_65 = None
    mul_151: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
    unsqueeze_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_97, -1);  view_97 = None
    mul_152: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_32);  mul_151 = unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_98: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_152, [384, 64, 3, 3]);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_44: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_149, view_98, arg98_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_149 = view_98 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_35: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_44)
    mul_153: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_44, sigmoid_35);  convolution_44 = sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_99: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg99_1, [1, 384, -1]);  arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_154: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg100_1, 0.07450538873672485);  arg100_1 = None
    view_100: "f32[384]" = torch.ops.aten.view.default(mul_154, [-1]);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [0, 2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 384, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 384, 1]" = var_mean_33[1];  var_mean_33 = None
    add_39: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_33: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_99, getitem_67);  view_99 = getitem_67 = None
    mul_155: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
    unsqueeze_33: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_100, -1);  view_100 = None
    mul_156: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_33);  mul_155 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_101: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_156, [384, 64, 3, 3]);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_45: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_153, view_101, arg101_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_153 = view_101 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_36: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_45)
    mul_157: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_45, sigmoid_36);  convolution_45 = sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_102: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg102_1, [1, 1536, -1]);  arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_158: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg103_1, 0.09125009274634042);  arg103_1 = None
    view_103: "f32[1536]" = torch.ops.aten.view.default(mul_158, [-1]);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [0, 2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1536, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 1536, 1]" = var_mean_34[1];  var_mean_34 = None
    add_40: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_34: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_102, getitem_69);  view_102 = getitem_69 = None
    mul_159: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
    unsqueeze_34: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_103, -1);  view_103 = None
    mul_160: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_34);  mul_159 = unsqueeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_104: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_160, [1536, 384, 1, 1]);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_46: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_157, view_104, arg104_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_157 = view_104 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_47: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_6, arg195_1, arg196_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg195_1 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_6: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_48: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_6, arg197_1, arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_6 = arg197_1 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_37: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_161: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_37);  convolution_46 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_162: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_161, 2.0);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_163: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_162, 0.2);  mul_162 = None
    add_41: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_163, add_36);  mul_163 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_38: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_41)
    mul_164: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_38);  sigmoid_38 = None
    mul_165: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_164, 0.9284766908852592);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_105: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg105_1, [1, 384, -1]);  arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_166: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg106_1, 0.04562504637317021);  arg106_1 = None
    view_106: "f32[384]" = torch.ops.aten.view.default(mul_166, [-1]);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [0, 2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 384, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 384, 1]" = var_mean_35[1];  var_mean_35 = None
    add_42: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_35: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_105, getitem_71);  view_105 = getitem_71 = None
    mul_167: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
    unsqueeze_35: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_106, -1);  view_106 = None
    mul_168: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_35);  mul_167 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_107: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_168, [384, 1536, 1, 1]);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_49: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_165, view_107, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_165 = view_107 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_39: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_49)
    mul_169: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_49, sigmoid_39);  convolution_49 = sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_108: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg108_1, [1, 384, -1]);  arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_170: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg109_1, 0.07450538873672485);  arg109_1 = None
    view_109: "f32[384]" = torch.ops.aten.view.default(mul_170, [-1]);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [0, 2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 384, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 384, 1]" = var_mean_36[1];  var_mean_36 = None
    add_43: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_36: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_108, getitem_73);  view_108 = getitem_73 = None
    mul_171: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
    unsqueeze_36: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_109, -1);  view_109 = None
    mul_172: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_36);  mul_171 = unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_110: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_172, [384, 64, 3, 3]);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_50: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_169, view_110, arg110_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_169 = view_110 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_40: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_50)
    mul_173: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_50, sigmoid_40);  convolution_50 = sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_111: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg111_1, [1, 384, -1]);  arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_174: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg112_1, 0.07450538873672485);  arg112_1 = None
    view_112: "f32[384]" = torch.ops.aten.view.default(mul_174, [-1]);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [0, 2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 384, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 384, 1]" = var_mean_37[1];  var_mean_37 = None
    add_44: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_37: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_111, getitem_75);  view_111 = getitem_75 = None
    mul_175: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
    unsqueeze_37: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_112, -1);  view_112 = None
    mul_176: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_37);  mul_175 = unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_113: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_176, [384, 64, 3, 3]);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_51: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_173, view_113, arg113_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_173 = view_113 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_41: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_51)
    mul_177: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_51, sigmoid_41);  convolution_51 = sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_114: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg114_1, [1, 1536, -1]);  arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg115_1, 0.09125009274634042);  arg115_1 = None
    view_115: "f32[1536]" = torch.ops.aten.view.default(mul_178, [-1]);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [0, 2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1536, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 1536, 1]" = var_mean_38[1];  var_mean_38 = None
    add_45: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_38: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_114, getitem_77);  view_114 = getitem_77 = None
    mul_179: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
    unsqueeze_38: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_115, -1);  view_115 = None
    mul_180: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_38);  mul_179 = unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_116: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_180, [1536, 384, 1, 1]);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_52: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_177, view_116, arg116_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_177 = view_116 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_52, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_53: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg199_1, arg200_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg199_1 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_7: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_54: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_7, arg201_1, arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg201_1 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_42: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_181: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_42);  convolution_52 = sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_182: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_181, 2.0);  mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_183: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_182, 0.2);  mul_182 = None
    add_46: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_183, add_41);  mul_183 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_43: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_46)
    mul_184: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_46, sigmoid_43);  sigmoid_43 = None
    mul_185: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_184, 0.9128709291752768);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_117: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg117_1, [1, 384, -1]);  arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_186: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg118_1, 0.04562504637317021);  arg118_1 = None
    view_118: "f32[384]" = torch.ops.aten.view.default(mul_186, [-1]);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [0, 2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 384, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 384, 1]" = var_mean_39[1];  var_mean_39 = None
    add_47: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_39: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_117, getitem_79);  view_117 = getitem_79 = None
    mul_187: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
    unsqueeze_39: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_118, -1);  view_118 = None
    mul_188: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_39);  mul_187 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_119: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_188, [384, 1536, 1, 1]);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_55: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_185, view_119, arg119_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_185 = view_119 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_44: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_55)
    mul_189: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_55, sigmoid_44);  convolution_55 = sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_120: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg120_1, [1, 384, -1]);  arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_190: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg121_1, 0.07450538873672485);  arg121_1 = None
    view_121: "f32[384]" = torch.ops.aten.view.default(mul_190, [-1]);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [0, 2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 384, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 384, 1]" = var_mean_40[1];  var_mean_40 = None
    add_48: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_40: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_120, getitem_81);  view_120 = getitem_81 = None
    mul_191: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
    unsqueeze_40: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_121, -1);  view_121 = None
    mul_192: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_40);  mul_191 = unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_122: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_192, [384, 64, 3, 3]);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_56: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_189, view_122, arg122_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_189 = view_122 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_45: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_56)
    mul_193: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_56, sigmoid_45);  convolution_56 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_123: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg123_1, [1, 384, -1]);  arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_194: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg124_1, 0.07450538873672485);  arg124_1 = None
    view_124: "f32[384]" = torch.ops.aten.view.default(mul_194, [-1]);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [0, 2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 384, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 384, 1]" = var_mean_41[1];  var_mean_41 = None
    add_49: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_41: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_123, getitem_83);  view_123 = getitem_83 = None
    mul_195: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
    unsqueeze_41: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_124, -1);  view_124 = None
    mul_196: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_41);  mul_195 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_125: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_196, [384, 64, 3, 3]);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_57: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_193, view_125, arg125_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_193 = view_125 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_46: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_57)
    mul_197: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_57, sigmoid_46);  convolution_57 = sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_126: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg126_1, [1, 1536, -1]);  arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_198: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg127_1, 0.09125009274634042);  arg127_1 = None
    view_127: "f32[1536]" = torch.ops.aten.view.default(mul_198, [-1]);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [0, 2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1536, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1536, 1]" = var_mean_42[1];  var_mean_42 = None
    add_50: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_42: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_126, getitem_85);  view_126 = getitem_85 = None
    mul_199: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
    unsqueeze_42: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_127, -1);  view_127 = None
    mul_200: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_42);  mul_199 = unsqueeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_128: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_200, [1536, 384, 1, 1]);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_58: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_197, view_128, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_197 = view_128 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_58, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_59: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg203_1, arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg203_1 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_8: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_60: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_8, arg205_1, arg206_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg205_1 = arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_47: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_201: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_47);  convolution_58 = sigmoid_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_202: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_201, 2.0);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_203: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_202, 0.2);  mul_202 = None
    add_51: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_203, add_46);  mul_203 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_48: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_51)
    mul_204: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_51, sigmoid_48);  add_51 = sigmoid_48 = None
    mul_205: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_204, 0.8980265101338745);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_2: "f32[8, 1536, 9, 9]" = torch.ops.aten.avg_pool2d.default(mul_205, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_129: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(arg129_1, [1, 1536, -1]);  arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_206: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg130_1, 0.04562504637317021);  arg130_1 = None
    view_130: "f32[1536]" = torch.ops.aten.view.default(mul_206, [-1]);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [0, 2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1536, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1536, 1]" = var_mean_43[1];  var_mean_43 = None
    add_52: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_43: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, getitem_87);  view_129 = getitem_87 = None
    mul_207: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
    unsqueeze_43: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_130, -1);  view_130 = None
    mul_208: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_43);  mul_207 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_131: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_208, [1536, 1536, 1, 1]);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_61: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(avg_pool2d_2, view_131, arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_2 = view_131 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_132: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg132_1, [1, 384, -1]);  arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_209: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg133_1, 0.04562504637317021);  arg133_1 = None
    view_133: "f32[384]" = torch.ops.aten.view.default(mul_209, [-1]);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [0, 2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 384, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 384, 1]" = var_mean_44[1];  var_mean_44 = None
    add_53: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_44: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_132, getitem_89);  view_132 = getitem_89 = None
    mul_210: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
    unsqueeze_44: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_133, -1);  view_133 = None
    mul_211: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_44);  mul_210 = unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_134: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_211, [384, 1536, 1, 1]);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_62: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_205, view_134, arg134_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_205 = view_134 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_49: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_62)
    mul_212: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_62, sigmoid_49);  convolution_62 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_135: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg135_1, [1, 384, -1]);  arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_213: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg136_1, 0.07450538873672485);  arg136_1 = None
    view_136: "f32[384]" = torch.ops.aten.view.default(mul_213, [-1]);  mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [0, 2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 384, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 384, 1]" = var_mean_45[1];  var_mean_45 = None
    add_54: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_45: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_135, getitem_91);  view_135 = getitem_91 = None
    mul_214: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
    unsqueeze_45: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_136, -1);  view_136 = None
    mul_215: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_45);  mul_214 = unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_137: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_215, [384, 64, 3, 3]);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_63: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_212, view_137, arg137_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  mul_212 = view_137 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_50: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_63)
    mul_216: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_63, sigmoid_50);  convolution_63 = sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_138: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg138_1, [1, 384, -1]);  arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_217: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg139_1, 0.07450538873672485);  arg139_1 = None
    view_139: "f32[384]" = torch.ops.aten.view.default(mul_217, [-1]);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [0, 2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 384, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 384, 1]" = var_mean_46[1];  var_mean_46 = None
    add_55: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_46: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_138, getitem_93);  view_138 = getitem_93 = None
    mul_218: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
    unsqueeze_46: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_139, -1);  view_139 = None
    mul_219: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_46);  mul_218 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_140: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_219, [384, 64, 3, 3]);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_64: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_216, view_140, arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_216 = view_140 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_51: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_64)
    mul_220: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_64, sigmoid_51);  convolution_64 = sigmoid_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_141: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg141_1, [1, 1536, -1]);  arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_221: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg142_1, 0.09125009274634042);  arg142_1 = None
    view_142: "f32[1536]" = torch.ops.aten.view.default(mul_221, [-1]);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [0, 2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 1536, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 1536, 1]" = var_mean_47[1];  var_mean_47 = None
    add_56: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_47: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_141, getitem_95);  view_141 = getitem_95 = None
    mul_222: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
    unsqueeze_47: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_142, -1);  view_142 = None
    mul_223: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_222, unsqueeze_47);  mul_222 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_143: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_223, [1536, 384, 1, 1]);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_65: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(mul_220, view_143, arg143_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_220 = view_143 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_65, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_66: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg207_1, arg208_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg207_1 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_9: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_67: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_9, arg209_1, arg210_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg209_1 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_52: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_224: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_52);  convolution_65 = sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_225: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_224, 2.0);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_226: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_225, 0.2);  mul_225 = None
    add_57: "f32[8, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_226, convolution_61);  mul_226 = convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_53: "f32[8, 1536, 9, 9]" = torch.ops.aten.sigmoid.default(add_57)
    mul_227: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(add_57, sigmoid_53);  sigmoid_53 = None
    mul_228: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_227, 0.9805806756909201);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_144: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg144_1, [1, 384, -1]);  arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_229: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg145_1, 0.04562504637317021);  arg145_1 = None
    view_145: "f32[384]" = torch.ops.aten.view.default(mul_229, [-1]);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [0, 2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 384, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 384, 1]" = var_mean_48[1];  var_mean_48 = None
    add_58: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_48: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_144, getitem_97);  view_144 = getitem_97 = None
    mul_230: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
    unsqueeze_48: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_145, -1);  view_145 = None
    mul_231: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_48);  mul_230 = unsqueeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_146: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_231, [384, 1536, 1, 1]);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_68: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_228, view_146, arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_228 = view_146 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_54: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_68)
    mul_232: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_68, sigmoid_54);  convolution_68 = sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_147: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg147_1, [1, 384, -1]);  arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_233: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg148_1, 0.07450538873672485);  arg148_1 = None
    view_148: "f32[384]" = torch.ops.aten.view.default(mul_233, [-1]);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [0, 2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 384, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 384, 1]" = var_mean_49[1];  var_mean_49 = None
    add_59: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_49: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_147, getitem_99);  view_147 = getitem_99 = None
    mul_234: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
    unsqueeze_49: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_148, -1);  view_148 = None
    mul_235: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_49);  mul_234 = unsqueeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_149: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_235, [384, 64, 3, 3]);  mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_69: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_232, view_149, arg149_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_232 = view_149 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_55: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_69)
    mul_236: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_69, sigmoid_55);  convolution_69 = sigmoid_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_150: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg150_1, [1, 384, -1]);  arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_237: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg151_1, 0.07450538873672485);  arg151_1 = None
    view_151: "f32[384]" = torch.ops.aten.view.default(mul_237, [-1]);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [0, 2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 384, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 384, 1]" = var_mean_50[1];  var_mean_50 = None
    add_60: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_50: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_150, getitem_101);  view_150 = getitem_101 = None
    mul_238: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
    unsqueeze_50: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_151, -1);  view_151 = None
    mul_239: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_50);  mul_238 = unsqueeze_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_152: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_239, [384, 64, 3, 3]);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_70: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_236, view_152, arg152_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_236 = view_152 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_56: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_70)
    mul_240: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_70, sigmoid_56);  convolution_70 = sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_153: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg153_1, [1, 1536, -1]);  arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg154_1, 0.09125009274634042);  arg154_1 = None
    view_154: "f32[1536]" = torch.ops.aten.view.default(mul_241, [-1]);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [0, 2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1536, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1536, 1]" = var_mean_51[1];  var_mean_51 = None
    add_61: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_51: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_153, getitem_103);  view_153 = getitem_103 = None
    mul_242: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
    unsqueeze_51: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_154, -1);  view_154 = None
    mul_243: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_51);  mul_242 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_155: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_243, [1536, 384, 1, 1]);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_71: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(mul_240, view_155, arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_240 = view_155 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_71, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_72: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg211_1, arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg211_1 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_10: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_73: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_10, arg213_1, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg213_1 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_57: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_244: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_57);  convolution_71 = sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_245: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_244, 2.0);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_246: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_245, 0.2);  mul_245 = None
    add_62: "f32[8, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_246, add_57);  mul_246 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_58: "f32[8, 1536, 9, 9]" = torch.ops.aten.sigmoid.default(add_62)
    mul_247: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(add_62, sigmoid_58);  sigmoid_58 = None
    mul_248: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_247, 0.9622504486493761);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_156: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg156_1, [1, 384, -1]);  arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_249: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg157_1, 0.04562504637317021);  arg157_1 = None
    view_157: "f32[384]" = torch.ops.aten.view.default(mul_249, [-1]);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [0, 2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 384, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 384, 1]" = var_mean_52[1];  var_mean_52 = None
    add_63: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_52: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_156, getitem_105);  view_156 = getitem_105 = None
    mul_250: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
    unsqueeze_52: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_157, -1);  view_157 = None
    mul_251: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_52);  mul_250 = unsqueeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_158: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_251, [384, 1536, 1, 1]);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_74: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_248, view_158, arg158_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_248 = view_158 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_59: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_74)
    mul_252: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_74, sigmoid_59);  convolution_74 = sigmoid_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_159: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg159_1, [1, 384, -1]);  arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_253: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg160_1, 0.07450538873672485);  arg160_1 = None
    view_160: "f32[384]" = torch.ops.aten.view.default(mul_253, [-1]);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [0, 2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 384, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 384, 1]" = var_mean_53[1];  var_mean_53 = None
    add_64: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_53: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_159, getitem_107);  view_159 = getitem_107 = None
    mul_254: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
    unsqueeze_53: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_160, -1);  view_160 = None
    mul_255: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_53);  mul_254 = unsqueeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_161: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_255, [384, 64, 3, 3]);  mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_75: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_252, view_161, arg161_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_252 = view_161 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_60: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_75)
    mul_256: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_75, sigmoid_60);  convolution_75 = sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_162: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg162_1, [1, 384, -1]);  arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_257: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg163_1, 0.07450538873672485);  arg163_1 = None
    view_163: "f32[384]" = torch.ops.aten.view.default(mul_257, [-1]);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [0, 2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 384, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 384, 1]" = var_mean_54[1];  var_mean_54 = None
    add_65: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_54: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = getitem_109 = None
    mul_258: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
    unsqueeze_54: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_163, -1);  view_163 = None
    mul_259: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_54);  mul_258 = unsqueeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_164: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_259, [384, 64, 3, 3]);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_76: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_256, view_164, arg164_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_256 = view_164 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_61: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_76)
    mul_260: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_76, sigmoid_61);  convolution_76 = sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_165: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg165_1, [1, 1536, -1]);  arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_261: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg166_1, 0.09125009274634042);  arg166_1 = None
    view_166: "f32[1536]" = torch.ops.aten.view.default(mul_261, [-1]);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [0, 2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1536, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 1536, 1]" = var_mean_55[1];  var_mean_55 = None
    add_66: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_55: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_165, getitem_111);  view_165 = getitem_111 = None
    mul_262: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
    unsqueeze_55: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_166, -1);  view_166 = None
    mul_263: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_55);  mul_262 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_167: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_263, [1536, 384, 1, 1]);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_77: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(mul_260, view_167, arg167_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_260 = view_167 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_78: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg215_1, arg216_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg215_1 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_11: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_78);  convolution_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_79: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_11, arg217_1, arg218_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg217_1 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_62: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_264: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_62);  convolution_77 = sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_265: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_264, 2.0);  mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_266: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_265, 0.2);  mul_265 = None
    add_67: "f32[8, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_266, add_62);  mul_266 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_168: "f32[1, 2304, 1536]" = torch.ops.aten.view.default(arg168_1, [1, 2304, -1]);  arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_267: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg169_1, 0.04562504637317021);  arg169_1 = None
    view_169: "f32[2304]" = torch.ops.aten.view.default(mul_267, [-1]);  mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [0, 2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 2304, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 2304, 1]" = var_mean_56[1];  var_mean_56 = None
    add_68: "f32[1, 2304, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 2304, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_56: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = getitem_113 = None
    mul_268: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
    unsqueeze_56: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(view_169, -1);  view_169 = None
    mul_269: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_56);  mul_268 = unsqueeze_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_170: "f32[2304, 1536, 1, 1]" = torch.ops.aten.view.default(mul_269, [2304, 1536, 1, 1]);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_80: "f32[8, 2304, 9, 9]" = torch.ops.aten.convolution.default(add_67, view_170, arg170_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_67 = view_170 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:445, code: x = self.final_act(x)
    sigmoid_63: "f32[8, 2304, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_80)
    mul_270: "f32[8, 2304, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_80, sigmoid_63);  convolution_80 = sigmoid_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_12: "f32[8, 2304, 1, 1]" = torch.ops.aten.mean.dim(mul_270, [-1, -2], True);  mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_171: "f32[8, 2304]" = torch.ops.aten.view.default(mean_12, [8, 2304]);  mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[8, 2304]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[2304, 1000]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg220_1, clone, permute);  arg220_1 = clone = permute = None
    return (addmm,)
    