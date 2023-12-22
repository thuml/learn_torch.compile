from __future__ import annotations



def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[16, 1, 1, 1]", primals_3: "f32[16]", primals_4: "f32[32, 16, 3, 3]", primals_5: "f32[32, 1, 1, 1]", primals_6: "f32[32]", primals_7: "f32[64, 32, 3, 3]", primals_8: "f32[64, 1, 1, 1]", primals_9: "f32[64]", primals_10: "f32[128, 64, 3, 3]", primals_11: "f32[128, 1, 1, 1]", primals_12: "f32[128]", primals_13: "f32[256, 128, 1, 1]", primals_14: "f32[256, 1, 1, 1]", primals_15: "f32[256]", primals_16: "f32[64, 128, 1, 1]", primals_17: "f32[64, 1, 1, 1]", primals_18: "f32[64]", primals_19: "f32[64, 64, 3, 3]", primals_20: "f32[64, 1, 1, 1]", primals_21: "f32[64]", primals_22: "f32[64, 64, 3, 3]", primals_23: "f32[64, 1, 1, 1]", primals_24: "f32[64]", primals_25: "f32[256, 64, 1, 1]", primals_26: "f32[256, 1, 1, 1]", primals_27: "f32[256]", primals_28: "f32[512, 256, 1, 1]", primals_29: "f32[512, 1, 1, 1]", primals_30: "f32[512]", primals_31: "f32[128, 256, 1, 1]", primals_32: "f32[128, 1, 1, 1]", primals_33: "f32[128]", primals_34: "f32[128, 64, 3, 3]", primals_35: "f32[128, 1, 1, 1]", primals_36: "f32[128]", primals_37: "f32[128, 64, 3, 3]", primals_38: "f32[128, 1, 1, 1]", primals_39: "f32[128]", primals_40: "f32[512, 128, 1, 1]", primals_41: "f32[512, 1, 1, 1]", primals_42: "f32[512]", primals_43: "f32[128, 512, 1, 1]", primals_44: "f32[128, 1, 1, 1]", primals_45: "f32[128]", primals_46: "f32[128, 64, 3, 3]", primals_47: "f32[128, 1, 1, 1]", primals_48: "f32[128]", primals_49: "f32[128, 64, 3, 3]", primals_50: "f32[128, 1, 1, 1]", primals_51: "f32[128]", primals_52: "f32[512, 128, 1, 1]", primals_53: "f32[512, 1, 1, 1]", primals_54: "f32[512]", primals_55: "f32[1536, 512, 1, 1]", primals_56: "f32[1536, 1, 1, 1]", primals_57: "f32[1536]", primals_58: "f32[384, 512, 1, 1]", primals_59: "f32[384, 1, 1, 1]", primals_60: "f32[384]", primals_61: "f32[384, 64, 3, 3]", primals_62: "f32[384, 1, 1, 1]", primals_63: "f32[384]", primals_64: "f32[384, 64, 3, 3]", primals_65: "f32[384, 1, 1, 1]", primals_66: "f32[384]", primals_67: "f32[1536, 384, 1, 1]", primals_68: "f32[1536, 1, 1, 1]", primals_69: "f32[1536]", primals_70: "f32[384, 1536, 1, 1]", primals_71: "f32[384, 1, 1, 1]", primals_72: "f32[384]", primals_73: "f32[384, 64, 3, 3]", primals_74: "f32[384, 1, 1, 1]", primals_75: "f32[384]", primals_76: "f32[384, 64, 3, 3]", primals_77: "f32[384, 1, 1, 1]", primals_78: "f32[384]", primals_79: "f32[1536, 384, 1, 1]", primals_80: "f32[1536, 1, 1, 1]", primals_81: "f32[1536]", primals_82: "f32[384, 1536, 1, 1]", primals_83: "f32[384, 1, 1, 1]", primals_84: "f32[384]", primals_85: "f32[384, 64, 3, 3]", primals_86: "f32[384, 1, 1, 1]", primals_87: "f32[384]", primals_88: "f32[384, 64, 3, 3]", primals_89: "f32[384, 1, 1, 1]", primals_90: "f32[384]", primals_91: "f32[1536, 384, 1, 1]", primals_92: "f32[1536, 1, 1, 1]", primals_93: "f32[1536]", primals_94: "f32[384, 1536, 1, 1]", primals_95: "f32[384, 1, 1, 1]", primals_96: "f32[384]", primals_97: "f32[384, 64, 3, 3]", primals_98: "f32[384, 1, 1, 1]", primals_99: "f32[384]", primals_100: "f32[384, 64, 3, 3]", primals_101: "f32[384, 1, 1, 1]", primals_102: "f32[384]", primals_103: "f32[1536, 384, 1, 1]", primals_104: "f32[1536, 1, 1, 1]", primals_105: "f32[1536]", primals_106: "f32[384, 1536, 1, 1]", primals_107: "f32[384, 1, 1, 1]", primals_108: "f32[384]", primals_109: "f32[384, 64, 3, 3]", primals_110: "f32[384, 1, 1, 1]", primals_111: "f32[384]", primals_112: "f32[384, 64, 3, 3]", primals_113: "f32[384, 1, 1, 1]", primals_114: "f32[384]", primals_115: "f32[1536, 384, 1, 1]", primals_116: "f32[1536, 1, 1, 1]", primals_117: "f32[1536]", primals_118: "f32[384, 1536, 1, 1]", primals_119: "f32[384, 1, 1, 1]", primals_120: "f32[384]", primals_121: "f32[384, 64, 3, 3]", primals_122: "f32[384, 1, 1, 1]", primals_123: "f32[384]", primals_124: "f32[384, 64, 3, 3]", primals_125: "f32[384, 1, 1, 1]", primals_126: "f32[384]", primals_127: "f32[1536, 384, 1, 1]", primals_128: "f32[1536, 1, 1, 1]", primals_129: "f32[1536]", primals_130: "f32[1536, 1536, 1, 1]", primals_131: "f32[1536, 1, 1, 1]", primals_132: "f32[1536]", primals_133: "f32[384, 1536, 1, 1]", primals_134: "f32[384, 1, 1, 1]", primals_135: "f32[384]", primals_136: "f32[384, 64, 3, 3]", primals_137: "f32[384, 1, 1, 1]", primals_138: "f32[384]", primals_139: "f32[384, 64, 3, 3]", primals_140: "f32[384, 1, 1, 1]", primals_141: "f32[384]", primals_142: "f32[1536, 384, 1, 1]", primals_143: "f32[1536, 1, 1, 1]", primals_144: "f32[1536]", primals_145: "f32[384, 1536, 1, 1]", primals_146: "f32[384, 1, 1, 1]", primals_147: "f32[384]", primals_148: "f32[384, 64, 3, 3]", primals_149: "f32[384, 1, 1, 1]", primals_150: "f32[384]", primals_151: "f32[384, 64, 3, 3]", primals_152: "f32[384, 1, 1, 1]", primals_153: "f32[384]", primals_154: "f32[1536, 384, 1, 1]", primals_155: "f32[1536, 1, 1, 1]", primals_156: "f32[1536]", primals_157: "f32[384, 1536, 1, 1]", primals_158: "f32[384, 1, 1, 1]", primals_159: "f32[384]", primals_160: "f32[384, 64, 3, 3]", primals_161: "f32[384, 1, 1, 1]", primals_162: "f32[384]", primals_163: "f32[384, 64, 3, 3]", primals_164: "f32[384, 1, 1, 1]", primals_165: "f32[384]", primals_166: "f32[1536, 384, 1, 1]", primals_167: "f32[1536, 1, 1, 1]", primals_168: "f32[1536]", primals_169: "f32[2304, 1536, 1, 1]", primals_170: "f32[2304, 1, 1, 1]", primals_171: "f32[2304]", primals_172: "f32[64, 256, 1, 1]", primals_173: "f32[64]", primals_174: "f32[256, 64, 1, 1]", primals_175: "f32[256]", primals_176: "f32[128, 512, 1, 1]", primals_177: "f32[128]", primals_178: "f32[512, 128, 1, 1]", primals_179: "f32[512]", primals_180: "f32[128, 512, 1, 1]", primals_181: "f32[128]", primals_182: "f32[512, 128, 1, 1]", primals_183: "f32[512]", primals_184: "f32[384, 1536, 1, 1]", primals_185: "f32[384]", primals_186: "f32[1536, 384, 1, 1]", primals_187: "f32[1536]", primals_188: "f32[384, 1536, 1, 1]", primals_189: "f32[384]", primals_190: "f32[1536, 384, 1, 1]", primals_191: "f32[1536]", primals_192: "f32[384, 1536, 1, 1]", primals_193: "f32[384]", primals_194: "f32[1536, 384, 1, 1]", primals_195: "f32[1536]", primals_196: "f32[384, 1536, 1, 1]", primals_197: "f32[384]", primals_198: "f32[1536, 384, 1, 1]", primals_199: "f32[1536]", primals_200: "f32[384, 1536, 1, 1]", primals_201: "f32[384]", primals_202: "f32[1536, 384, 1, 1]", primals_203: "f32[1536]", primals_204: "f32[384, 1536, 1, 1]", primals_205: "f32[384]", primals_206: "f32[1536, 384, 1, 1]", primals_207: "f32[1536]", primals_208: "f32[384, 1536, 1, 1]", primals_209: "f32[384]", primals_210: "f32[1536, 384, 1, 1]", primals_211: "f32[1536]", primals_212: "f32[384, 1536, 1, 1]", primals_213: "f32[384]", primals_214: "f32[1536, 384, 1, 1]", primals_215: "f32[1536]", primals_216: "f32[384, 1536, 1, 1]", primals_217: "f32[384]", primals_218: "f32[1536, 384, 1, 1]", primals_219: "f32[1536]", primals_220: "f32[1000, 2304]", primals_221: "f32[1000]", primals_222: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view: "f32[1, 16, 27]" = torch.ops.aten.view.default(primals_1, [1, 16, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_2, 0.34412564994580647)
    view_1: "f32[16]" = torch.ops.aten.view.default(mul, [-1]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(view, [0, 2], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
    mul_1: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2]);  rsqrt = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
    mul_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze);  mul_1 = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_2: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_2, [16, 3, 3, 3]);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_222, view_2, primals_3, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid: "f32[8, 16, 112, 112]" = torch.ops.aten.sigmoid.default(convolution)
    mul_3: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(convolution, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_3: "f32[1, 32, 144]" = torch.ops.aten.view.default(primals_4, [1, 32, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_4: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_5, 0.1490107774734497)
    view_4: "f32[32]" = torch.ops.aten.view.default(mul_4, [-1]);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [0, 2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1]" = var_mean_1[1];  var_mean_1 = None
    add_1: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub_1: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = None
    mul_5: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2]);  getitem_3 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2]);  rsqrt_1 = None
    unsqueeze_1: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_4, -1);  view_4 = None
    mul_6: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_1);  mul_5 = unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_5: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_6, [32, 16, 3, 3]);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_3, view_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(convolution_1)
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(convolution_1, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_6: "f32[1, 64, 288]" = torch.ops.aten.view.default(primals_7, [1, 64, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_8: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_8, 0.10536653122135592)
    view_7: "f32[64]" = torch.ops.aten.view.default(mul_8, [-1]);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [0, 2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None
    add_2: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_2: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = None
    mul_9: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2]);  getitem_5 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2]);  rsqrt_2 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_7, -1);  view_7 = None
    mul_10: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_2);  mul_9 = unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_8: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_10, [64, 32, 3, 3]);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_2: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(mul_7, view_8, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_2: "f32[8, 64, 112, 112]" = torch.ops.aten.sigmoid.default(convolution_2)
    mul_11: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(convolution_2, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_9: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_10, [1, 128, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_12: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_11, 0.07450538873672485)
    view_10: "f32[128]" = torch.ops.aten.view.default(mul_12, [-1]);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [0, 2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_3: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, getitem_7);  view_9 = None
    mul_13: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2]);  getitem_7 = None
    squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2]);  rsqrt_3 = None
    unsqueeze_3: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_10, -1);  view_10 = None
    mul_14: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_3);  mul_13 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_11: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_14, [128, 64, 3, 3]);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_3: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(mul_11, view_11, primals_12, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_3: "f32[8, 128, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_3)
    mul_15: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, sigmoid_3);  sigmoid_3 = None
    mul_16: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_15, 1.0);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_12: "f32[1, 256, 128]" = torch.ops.aten.view.default(primals_13, [1, 256, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_17: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_14, 0.1580497968320339)
    view_13: "f32[256]" = torch.ops.aten.view.default(mul_17, [-1]);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [0, 2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 256, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 256, 1]" = var_mean_4[1];  var_mean_4 = None
    add_4: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_4: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = None
    mul_18: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_8: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2]);  getitem_9 = None
    squeeze_9: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2]);  rsqrt_4 = None
    unsqueeze_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_13, -1);  view_13 = None
    mul_19: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_4);  mul_18 = unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_14: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_19, [256, 128, 1, 1]);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_4: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(mul_16, view_14, primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_15: "f32[1, 64, 128]" = torch.ops.aten.view.default(primals_16, [1, 64, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_20: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_17, 0.1580497968320339)
    view_16: "f32[64]" = torch.ops.aten.view.default(mul_20, [-1]);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0, 2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 64, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 64, 1]" = var_mean_5[1];  var_mean_5 = None
    add_5: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_5: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = None
    mul_21: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2]);  getitem_11 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2]);  rsqrt_5 = None
    unsqueeze_5: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_16, -1);  view_16 = None
    mul_22: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_5);  mul_21 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_17: "f32[64, 128, 1, 1]" = torch.ops.aten.view.default(mul_22, [64, 128, 1, 1]);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(mul_16, view_17, primals_18, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_4: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_5)
    mul_23: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_18: "f32[1, 64, 576]" = torch.ops.aten.view.default(primals_19, [1, 64, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_24: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_20, 0.07450538873672485)
    view_19: "f32[64]" = torch.ops.aten.view.default(mul_24, [-1]);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [0, 2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 64, 1]" = var_mean_6[1];  var_mean_6 = None
    add_6: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_6: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_18, getitem_13);  view_18 = None
    mul_25: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2]);  getitem_13 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2]);  rsqrt_6 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_19, -1);  view_19 = None
    mul_26: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_6);  mul_25 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_20: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_26, [64, 64, 3, 3]);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_6: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(mul_23, view_20, primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_5: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_6)
    mul_27: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_6, sigmoid_5);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_21: "f32[1, 64, 576]" = torch.ops.aten.view.default(primals_22, [1, 64, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_28: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_23, 0.07450538873672485)
    view_22: "f32[64]" = torch.ops.aten.view.default(mul_28, [-1]);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [0, 2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 64, 1]" = var_mean_7[1];  var_mean_7 = None
    add_7: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_7: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_21, getitem_15);  view_21 = None
    mul_29: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2]);  getitem_15 = None
    squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2]);  rsqrt_7 = None
    unsqueeze_7: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_22, -1);  view_22 = None
    mul_30: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_7);  mul_29 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_23: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_30, [64, 64, 3, 3]);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_7: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(mul_27, view_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_6: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_7)
    mul_31: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_24: "f32[1, 256, 64]" = torch.ops.aten.view.default(primals_25, [1, 256, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_32: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_26, 0.22351616621017456)
    view_25: "f32[256]" = torch.ops.aten.view.default(mul_32, [-1]);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [0, 2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 256, 1]" = var_mean_8[1];  var_mean_8 = None
    add_8: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_8: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_24, getitem_17);  view_24 = None
    mul_33: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2]);  getitem_17 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2]);  rsqrt_8 = None
    unsqueeze_8: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_25, -1);  view_25 = None
    mul_34: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_8);  mul_33 = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_26: "f32[256, 64, 1, 1]" = torch.ops.aten.view.default(mul_34, [256, 64, 1, 1]);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_8: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(mul_31, view_26, primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_8, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_9: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_172, primals_173, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_10: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu, primals_174, primals_175, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_35: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid_7);  sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_36: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, 2.0);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_37: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_36, 0.2);  mul_36 = None
    add_9: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_37, convolution_4);  mul_37 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_8: "f32[8, 256, 56, 56]" = torch.ops.aten.sigmoid.default(add_9)
    mul_38: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_8)
    mul_39: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_38, 0.9805806756909201);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d: "f32[8, 256, 28, 28]" = torch.ops.aten.avg_pool2d.default(mul_39, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_28, [1, 512, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_40: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_29, 0.11175808310508728)
    view_28: "f32[512]" = torch.ops.aten.view.default(mul_40, [-1]);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [0, 2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_19);  view_27 = None
    mul_41: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_18: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2]);  getitem_19 = None
    squeeze_19: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2]);  rsqrt_9 = None
    unsqueeze_9: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_28, -1);  view_28 = None
    mul_42: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_9);  mul_41 = unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_29: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_42, [512, 256, 1, 1]);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_11: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(avg_pool2d, view_29, primals_30, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_30: "f32[1, 128, 256]" = torch.ops.aten.view.default(primals_31, [1, 128, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_43: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_32, 0.11175808310508728)
    view_31: "f32[128]" = torch.ops.aten.view.default(mul_43, [-1]);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [0, 2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_10: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = None
    mul_44: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_20: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2]);  getitem_21 = None
    squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2]);  rsqrt_10 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_31, -1);  view_31 = None
    mul_45: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_10);  mul_44 = unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_32: "f32[128, 256, 1, 1]" = torch.ops.aten.view.default(mul_45, [128, 256, 1, 1]);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_12: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(mul_39, view_32, primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_9: "f32[8, 128, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_12)
    mul_46: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_12, sigmoid_9);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_33: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_34, [1, 128, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_47: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_35, 0.07450538873672485)
    view_34: "f32[128]" = torch.ops.aten.view.default(mul_47, [-1]);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [0, 2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_11: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_33, getitem_23);  view_33 = None
    mul_48: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2]);  getitem_23 = None
    squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2]);  rsqrt_11 = None
    unsqueeze_11: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_34, -1);  view_34 = None
    mul_49: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_11);  mul_48 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_35: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_49, [128, 64, 3, 3]);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_13: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_46, view_35, primals_36, [2, 2], [1, 1], [1, 1], False, [0, 0], 2);  primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_10: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_13)
    mul_50: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_13, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_36: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_37, [1, 128, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_51: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_38, 0.07450538873672485)
    view_37: "f32[128]" = torch.ops.aten.view.default(mul_51, [-1]);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [0, 2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_13: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_12: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_36, getitem_25);  view_36 = None
    mul_52: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_24: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2]);  getitem_25 = None
    squeeze_25: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2]);  rsqrt_12 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_37, -1);  view_37 = None
    mul_53: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_12);  mul_52 = unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_38: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_53, [128, 64, 3, 3]);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_14: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_50, view_38, primals_39, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_11: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_14)
    mul_54: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, sigmoid_11);  sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_39: "f32[1, 512, 128]" = torch.ops.aten.view.default(primals_40, [1, 512, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_55: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_41, 0.1580497968320339)
    view_40: "f32[512]" = torch.ops.aten.view.default(mul_55, [-1]);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [0, 2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_13: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_39, getitem_27);  view_39 = None
    mul_56: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_26: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2]);  getitem_27 = None
    squeeze_27: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2]);  rsqrt_13 = None
    unsqueeze_13: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_40, -1);  view_40 = None
    mul_57: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_13);  mul_56 = unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_41: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_57, [512, 128, 1, 1]);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_15: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(mul_54, view_41, primals_42, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_15, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_16: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_176, primals_177, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_1: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_17: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_1, primals_178, primals_179, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_58: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_59: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, 2.0);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_60: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, 0.2);  mul_59 = None
    add_15: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_60, convolution_11);  mul_60 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_13: "f32[8, 512, 28, 28]" = torch.ops.aten.sigmoid.default(add_15)
    mul_61: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_15, sigmoid_13)
    mul_62: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, 0.9805806756909201);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_42: "f32[1, 128, 512]" = torch.ops.aten.view.default(primals_43, [1, 128, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_63: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_44, 0.07902489841601695)
    view_43: "f32[128]" = torch.ops.aten.view.default(mul_63, [-1]);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [0, 2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_16: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_14: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_42, getitem_29);  view_42 = None
    mul_64: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_28: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2]);  getitem_29 = None
    squeeze_29: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2]);  rsqrt_14 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_43, -1);  view_43 = None
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_14);  mul_64 = unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_44: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(mul_65, [128, 512, 1, 1]);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_18: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_62, view_44, primals_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_14: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_18)
    mul_66: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_45: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_46, [1, 128, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_67: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_47, 0.07450538873672485)
    view_46: "f32[128]" = torch.ops.aten.view.default(mul_67, [-1]);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [0, 2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_15: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_45, getitem_31);  view_45 = None
    mul_68: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2]);  getitem_31 = None
    squeeze_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2]);  rsqrt_15 = None
    unsqueeze_15: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_46, -1);  view_46 = None
    mul_69: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_15);  mul_68 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_47: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_69, [128, 64, 3, 3]);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_19: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_66, view_47, primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_15: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_19)
    mul_70: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_19, sigmoid_15);  sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_48: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_49, [1, 128, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_71: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_50, 0.07450538873672485)
    view_49: "f32[128]" = torch.ops.aten.view.default(mul_71, [-1]);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [0, 2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_16: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_48, getitem_33);  view_48 = None
    mul_72: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2]);  getitem_33 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2]);  rsqrt_16 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_49, -1);  view_49 = None
    mul_73: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_16);  mul_72 = unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_50: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_73, [128, 64, 3, 3]);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_20: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_70, view_50, primals_51, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_16: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_20)
    mul_74: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_51: "f32[1, 512, 128]" = torch.ops.aten.view.default(primals_52, [1, 512, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_75: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_53, 0.1580497968320339)
    view_52: "f32[512]" = torch.ops.aten.view.default(mul_75, [-1]);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [0, 2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_19: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_17: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_51, getitem_35);  view_51 = None
    mul_76: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_34: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2]);  getitem_35 = None
    squeeze_35: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2]);  rsqrt_17 = None
    unsqueeze_17: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_52, -1);  view_52 = None
    mul_77: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_17);  mul_76 = unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_53: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_77, [512, 128, 1, 1]);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_21: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(mul_74, view_53, primals_54, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_21, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_22: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_2: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_23: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_2, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_78: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_17);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_79: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_78, 2.0);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_80: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, 0.2);  mul_79 = None
    add_20: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_80, add_15);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_18: "f32[8, 512, 28, 28]" = torch.ops.aten.sigmoid.default(add_20)
    mul_81: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_20, sigmoid_18)
    mul_82: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_81, 0.9622504486493761);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_1: "f32[8, 512, 14, 14]" = torch.ops.aten.avg_pool2d.default(mul_82, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_54: "f32[1, 1536, 512]" = torch.ops.aten.view.default(primals_55, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_83: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_56, 0.07902489841601695)
    view_55: "f32[1536]" = torch.ops.aten.view.default(mul_83, [-1]);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [0, 2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1536, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1536, 1]" = var_mean_18[1];  var_mean_18 = None
    add_21: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_18: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, getitem_37);  view_54 = None
    mul_84: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_36: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2]);  getitem_37 = None
    squeeze_37: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2]);  rsqrt_18 = None
    unsqueeze_18: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_55, -1);  view_55 = None
    mul_85: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_18);  mul_84 = unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_56: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_85, [1536, 512, 1, 1]);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(avg_pool2d_1, view_56, primals_57, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_57: "f32[1, 384, 512]" = torch.ops.aten.view.default(primals_58, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_86: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_59, 0.07902489841601695)
    view_58: "f32[384]" = torch.ops.aten.view.default(mul_86, [-1]);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [0, 2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 384, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 384, 1]" = var_mean_19[1];  var_mean_19 = None
    add_22: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_19: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_57, getitem_39);  view_57 = None
    mul_87: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_38: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2]);  getitem_39 = None
    squeeze_39: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2]);  rsqrt_19 = None
    unsqueeze_19: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_58, -1);  view_58 = None
    mul_88: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_19);  mul_87 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_59: "f32[384, 512, 1, 1]" = torch.ops.aten.view.default(mul_88, [384, 512, 1, 1]);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_25: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_82, view_59, primals_60, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_19: "f32[8, 384, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_25)
    mul_89: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_25, sigmoid_19);  sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_60: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_61, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_90: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_62, 0.07450538873672485)
    view_61: "f32[384]" = torch.ops.aten.view.default(mul_90, [-1]);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [0, 2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 384, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 384, 1]" = var_mean_20[1];  var_mean_20 = None
    add_23: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_20: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_60, getitem_41);  view_60 = None
    mul_91: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_40: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2]);  getitem_41 = None
    squeeze_41: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2]);  rsqrt_20 = None
    unsqueeze_20: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_61, -1);  view_61 = None
    mul_92: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_20);  mul_91 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_62: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_92, [384, 64, 3, 3]);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_26: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_89, view_62, primals_63, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_20: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_26)
    mul_93: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_63: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_64, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_94: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_65, 0.07450538873672485)
    view_64: "f32[384]" = torch.ops.aten.view.default(mul_94, [-1]);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [0, 2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 384, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 384, 1]" = var_mean_21[1];  var_mean_21 = None
    add_24: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_21: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_63, getitem_43);  view_63 = None
    mul_95: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_42: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2]);  getitem_43 = None
    squeeze_43: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2]);  rsqrt_21 = None
    unsqueeze_21: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_64, -1);  view_64 = None
    mul_96: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_21);  mul_95 = unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_65: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_96, [384, 64, 3, 3]);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_27: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_93, view_65, primals_66, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_21: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_97: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_66: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_67, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_98: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_68, 0.09125009274634042)
    view_67: "f32[1536]" = torch.ops.aten.view.default(mul_98, [-1]);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [0, 2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1536, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1536, 1]" = var_mean_22[1];  var_mean_22 = None
    add_25: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_22: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_66, getitem_45);  view_66 = None
    mul_99: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_44: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2]);  getitem_45 = None
    squeeze_45: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2]);  rsqrt_22 = None
    unsqueeze_22: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_67, -1);  view_67 = None
    mul_100: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_22);  mul_99 = unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_68: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_100, [1536, 384, 1, 1]);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(mul_97, view_68, primals_69, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_28, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_29: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_184, primals_185, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_30: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_186, primals_187, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_22: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_101: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_102: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_101, 2.0);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_103: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_102, 0.2);  mul_102 = None
    add_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_103, convolution_24);  mul_103 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_23: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_26)
    mul_104: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_26, sigmoid_23)
    mul_105: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_104, 0.9805806756909201);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_69: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_70, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_106: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_71, 0.04562504637317021)
    view_70: "f32[384]" = torch.ops.aten.view.default(mul_106, [-1]);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [0, 2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 384, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 384, 1]" = var_mean_23[1];  var_mean_23 = None
    add_27: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_23: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_69, getitem_47);  view_69 = None
    mul_107: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_46: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2]);  getitem_47 = None
    squeeze_47: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2]);  rsqrt_23 = None
    unsqueeze_23: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_70, -1);  view_70 = None
    mul_108: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_23);  mul_107 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_71: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_108, [384, 1536, 1, 1]);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_31: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_105, view_71, primals_72, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_24: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_31)
    mul_109: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, sigmoid_24);  sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_72: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_73, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_110: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_74, 0.07450538873672485)
    view_73: "f32[384]" = torch.ops.aten.view.default(mul_110, [-1]);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [0, 2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 384, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 384, 1]" = var_mean_24[1];  var_mean_24 = None
    add_28: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_24: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_72, getitem_49);  view_72 = None
    mul_111: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_48: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2]);  getitem_49 = None
    squeeze_49: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2]);  rsqrt_24 = None
    unsqueeze_24: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_73, -1);  view_73 = None
    mul_112: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_24);  mul_111 = unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_74: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_112, [384, 64, 3, 3]);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_32: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_109, view_74, primals_75, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_25: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_32)
    mul_113: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_32, sigmoid_25);  sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_75: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_76, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_114: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_77, 0.07450538873672485)
    view_76: "f32[384]" = torch.ops.aten.view.default(mul_114, [-1]);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [0, 2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 384, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 384, 1]" = var_mean_25[1];  var_mean_25 = None
    add_29: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_25: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_75, getitem_51);  view_75 = None
    mul_115: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_50: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2]);  getitem_51 = None
    squeeze_51: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2]);  rsqrt_25 = None
    unsqueeze_25: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_76, -1);  view_76 = None
    mul_116: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_25);  mul_115 = unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_77: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_116, [384, 64, 3, 3]);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_33: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_113, view_77, primals_78, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_26: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_33)
    mul_117: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_78: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_79, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_118: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_80, 0.09125009274634042)
    view_79: "f32[1536]" = torch.ops.aten.view.default(mul_118, [-1]);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [0, 2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 1536, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 1536, 1]" = var_mean_26[1];  var_mean_26 = None
    add_30: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_26: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_78, getitem_53);  view_78 = None
    mul_119: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_52: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2]);  getitem_53 = None
    squeeze_53: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2]);  rsqrt_26 = None
    unsqueeze_26: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_79, -1);  view_79 = None
    mul_120: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_26);  mul_119 = unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_80: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_120, [1536, 384, 1, 1]);  mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_34: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(mul_117, view_80, primals_81, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_35: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_188, primals_189, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_4: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_36: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_4, primals_190, primals_191, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_27: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_121: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_27);  sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_122: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, 2.0);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_123: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_122, 0.2);  mul_122 = None
    add_31: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_123, add_26);  mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_31)
    mul_124: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_31, sigmoid_28)
    mul_125: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, 0.9622504486493761);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_81: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_82, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_126: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_83, 0.04562504637317021)
    view_82: "f32[384]" = torch.ops.aten.view.default(mul_126, [-1]);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [0, 2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 384, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 384, 1]" = var_mean_27[1];  var_mean_27 = None
    add_32: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_27: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_81, getitem_55);  view_81 = None
    mul_127: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_54: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2]);  getitem_55 = None
    squeeze_55: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2]);  rsqrt_27 = None
    unsqueeze_27: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_82, -1);  view_82 = None
    mul_128: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_27);  mul_127 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_83: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_128, [384, 1536, 1, 1]);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_37: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_125, view_83, primals_84, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_29: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_37)
    mul_129: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_84: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_85, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_130: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_86, 0.07450538873672485)
    view_85: "f32[384]" = torch.ops.aten.view.default(mul_130, [-1]);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [0, 2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 384, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 384, 1]" = var_mean_28[1];  var_mean_28 = None
    add_33: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_28: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_84, getitem_57);  view_84 = None
    mul_131: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_56: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2]);  getitem_57 = None
    squeeze_57: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2]);  rsqrt_28 = None
    unsqueeze_28: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_85, -1);  view_85 = None
    mul_132: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_28);  mul_131 = unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_86: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_132, [384, 64, 3, 3]);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_38: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_129, view_86, primals_87, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_30: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_38)
    mul_133: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, sigmoid_30);  sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_87: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_88, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_134: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_89, 0.07450538873672485)
    view_88: "f32[384]" = torch.ops.aten.view.default(mul_134, [-1]);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [0, 2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 384, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 384, 1]" = var_mean_29[1];  var_mean_29 = None
    add_34: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_29: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_87, getitem_59);  view_87 = None
    mul_135: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_58: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2]);  getitem_59 = None
    squeeze_59: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2]);  rsqrt_29 = None
    unsqueeze_29: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_88, -1);  view_88 = None
    mul_136: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_29);  mul_135 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_89: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_136, [384, 64, 3, 3]);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_39: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_133, view_89, primals_90, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_31: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_39)
    mul_137: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, sigmoid_31);  sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_90: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_91, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_138: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_92, 0.09125009274634042)
    view_91: "f32[1536]" = torch.ops.aten.view.default(mul_138, [-1]);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [0, 2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 1536, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 1536, 1]" = var_mean_30[1];  var_mean_30 = None
    add_35: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_30: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_90, getitem_61);  view_90 = None
    mul_139: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_60: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2]);  getitem_61 = None
    squeeze_61: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2]);  rsqrt_30 = None
    unsqueeze_30: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_91, -1);  view_91 = None
    mul_140: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_30);  mul_139 = unsqueeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_92: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_140, [1536, 384, 1, 1]);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_40: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(mul_137, view_92, primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_40, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_41: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_192, primals_193, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_5: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_42: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_5, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_32: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_141: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_32);  sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_142: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_141, 2.0);  mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_143: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_142, 0.2);  mul_142 = None
    add_36: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, add_31);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_33: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_36)
    mul_144: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_36, sigmoid_33)
    mul_145: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_144, 0.9449111825230679);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_93: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_94, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_146: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_95, 0.04562504637317021)
    view_94: "f32[384]" = torch.ops.aten.view.default(mul_146, [-1]);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [0, 2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 384, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 384, 1]" = var_mean_31[1];  var_mean_31 = None
    add_37: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_31: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_93, getitem_63);  view_93 = None
    mul_147: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_62: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2]);  getitem_63 = None
    squeeze_63: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2]);  rsqrt_31 = None
    unsqueeze_31: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_94, -1);  view_94 = None
    mul_148: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_31);  mul_147 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_95: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_148, [384, 1536, 1, 1]);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_43: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_145, view_95, primals_96, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_34: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_149: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, sigmoid_34);  sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_96: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_97, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_150: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_98, 0.07450538873672485)
    view_97: "f32[384]" = torch.ops.aten.view.default(mul_150, [-1]);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [0, 2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 384, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 384, 1]" = var_mean_32[1];  var_mean_32 = None
    add_38: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_32: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_96, getitem_65);  view_96 = None
    mul_151: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_64: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2]);  getitem_65 = None
    squeeze_65: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2]);  rsqrt_32 = None
    unsqueeze_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_97, -1);  view_97 = None
    mul_152: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_32);  mul_151 = unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_98: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_152, [384, 64, 3, 3]);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_44: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_149, view_98, primals_99, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_35: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_44)
    mul_153: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_44, sigmoid_35);  sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_99: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_100, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_154: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_101, 0.07450538873672485)
    view_100: "f32[384]" = torch.ops.aten.view.default(mul_154, [-1]);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [0, 2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 384, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 384, 1]" = var_mean_33[1];  var_mean_33 = None
    add_39: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_33: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_99, getitem_67);  view_99 = None
    mul_155: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_66: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2]);  getitem_67 = None
    squeeze_67: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2]);  rsqrt_33 = None
    unsqueeze_33: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_100, -1);  view_100 = None
    mul_156: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_33);  mul_155 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_101: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_156, [384, 64, 3, 3]);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_45: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_153, view_101, primals_102, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_36: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_45)
    mul_157: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, sigmoid_36);  sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_102: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_103, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_158: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_104, 0.09125009274634042)
    view_103: "f32[1536]" = torch.ops.aten.view.default(mul_158, [-1]);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [0, 2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1536, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 1536, 1]" = var_mean_34[1];  var_mean_34 = None
    add_40: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_34: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_102, getitem_69);  view_102 = None
    mul_159: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_68: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2]);  getitem_69 = None
    squeeze_69: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2]);  rsqrt_34 = None
    unsqueeze_34: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_103, -1);  view_103 = None
    mul_160: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_34);  mul_159 = unsqueeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_104: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_160, [1536, 384, 1, 1]);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(mul_157, view_104, primals_105, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_47: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_196, primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_6: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_48: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_6, primals_198, primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_37: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_161: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_37);  sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_162: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_161, 2.0);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_163: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_162, 0.2);  mul_162 = None
    add_41: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_163, add_36);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_41)
    mul_164: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_38)
    mul_165: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_164, 0.9284766908852592);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_105: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_106, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_166: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_107, 0.04562504637317021)
    view_106: "f32[384]" = torch.ops.aten.view.default(mul_166, [-1]);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [0, 2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 384, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 384, 1]" = var_mean_35[1];  var_mean_35 = None
    add_42: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_35: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_105, getitem_71);  view_105 = None
    mul_167: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_70: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2]);  getitem_71 = None
    squeeze_71: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2]);  rsqrt_35 = None
    unsqueeze_35: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_106, -1);  view_106 = None
    mul_168: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_35);  mul_167 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_107: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_168, [384, 1536, 1, 1]);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_49: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_165, view_107, primals_108, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_39: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_49)
    mul_169: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, sigmoid_39);  sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_108: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_109, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_170: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_110, 0.07450538873672485)
    view_109: "f32[384]" = torch.ops.aten.view.default(mul_170, [-1]);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [0, 2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 384, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 384, 1]" = var_mean_36[1];  var_mean_36 = None
    add_43: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_36: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_108, getitem_73);  view_108 = None
    mul_171: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_72: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2]);  getitem_73 = None
    squeeze_73: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2]);  rsqrt_36 = None
    unsqueeze_36: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_109, -1);  view_109 = None
    mul_172: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_36);  mul_171 = unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_110: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_172, [384, 64, 3, 3]);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_50: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_169, view_110, primals_111, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_40: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_50)
    mul_173: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_50, sigmoid_40);  sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_111: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_112, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_174: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_113, 0.07450538873672485)
    view_112: "f32[384]" = torch.ops.aten.view.default(mul_174, [-1]);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [0, 2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 384, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 384, 1]" = var_mean_37[1];  var_mean_37 = None
    add_44: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_37: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_111, getitem_75);  view_111 = None
    mul_175: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_74: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2]);  getitem_75 = None
    squeeze_75: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2]);  rsqrt_37 = None
    unsqueeze_37: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_112, -1);  view_112 = None
    mul_176: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_37);  mul_175 = unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_113: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_176, [384, 64, 3, 3]);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_51: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_173, view_113, primals_114, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_41: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_51)
    mul_177: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, sigmoid_41);  sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_114: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_115, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_116, 0.09125009274634042)
    view_115: "f32[1536]" = torch.ops.aten.view.default(mul_178, [-1]);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [0, 2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1536, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 1536, 1]" = var_mean_38[1];  var_mean_38 = None
    add_45: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_38: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_114, getitem_77);  view_114 = None
    mul_179: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_76: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2]);  getitem_77 = None
    squeeze_77: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2]);  rsqrt_38 = None
    unsqueeze_38: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_115, -1);  view_115 = None
    mul_180: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_38);  mul_179 = unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_116: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_180, [1536, 384, 1, 1]);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_52: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(mul_177, view_116, primals_117, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_52, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_53: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_200, primals_201, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_7: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_54: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_202, primals_203, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_42: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_181: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_42);  sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_182: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, 2.0);  mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_183: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, 0.2);  mul_182 = None
    add_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_183, add_41);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_43: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_46)
    mul_184: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, sigmoid_43)
    mul_185: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, 0.9128709291752768);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_117: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_118, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_186: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_119, 0.04562504637317021)
    view_118: "f32[384]" = torch.ops.aten.view.default(mul_186, [-1]);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [0, 2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 384, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 384, 1]" = var_mean_39[1];  var_mean_39 = None
    add_47: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_39: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_117, getitem_79);  view_117 = None
    mul_187: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_78: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2]);  getitem_79 = None
    squeeze_79: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2]);  rsqrt_39 = None
    unsqueeze_39: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_118, -1);  view_118 = None
    mul_188: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_39);  mul_187 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_119: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_188, [384, 1536, 1, 1]);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_55: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_185, view_119, primals_120, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_44: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_55)
    mul_189: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, sigmoid_44);  sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_120: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_121, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_190: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_122, 0.07450538873672485)
    view_121: "f32[384]" = torch.ops.aten.view.default(mul_190, [-1]);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [0, 2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 384, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 384, 1]" = var_mean_40[1];  var_mean_40 = None
    add_48: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_40: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_120, getitem_81);  view_120 = None
    mul_191: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_80: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2]);  getitem_81 = None
    squeeze_81: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2]);  rsqrt_40 = None
    unsqueeze_40: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_121, -1);  view_121 = None
    mul_192: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_40);  mul_191 = unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_122: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_192, [384, 64, 3, 3]);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_56: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_189, view_122, primals_123, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_45: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_56)
    mul_193: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_56, sigmoid_45);  sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_123: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_124, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_194: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_125, 0.07450538873672485)
    view_124: "f32[384]" = torch.ops.aten.view.default(mul_194, [-1]);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [0, 2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 384, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 384, 1]" = var_mean_41[1];  var_mean_41 = None
    add_49: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_41: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_123, getitem_83);  view_123 = None
    mul_195: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_82: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2]);  getitem_83 = None
    squeeze_83: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2]);  rsqrt_41 = None
    unsqueeze_41: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_124, -1);  view_124 = None
    mul_196: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_41);  mul_195 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_125: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_196, [384, 64, 3, 3]);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_57: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_193, view_125, primals_126, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_46: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_57)
    mul_197: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, sigmoid_46);  sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_126: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_127, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_198: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_128, 0.09125009274634042)
    view_127: "f32[1536]" = torch.ops.aten.view.default(mul_198, [-1]);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [0, 2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1536, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1536, 1]" = var_mean_42[1];  var_mean_42 = None
    add_50: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_42: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_126, getitem_85);  view_126 = None
    mul_199: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_84: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2]);  getitem_85 = None
    squeeze_85: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2]);  rsqrt_42 = None
    unsqueeze_42: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_127, -1);  view_127 = None
    mul_200: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_42);  mul_199 = unsqueeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_128: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_200, [1536, 384, 1, 1]);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_58: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(mul_197, view_128, primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_58, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_59: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_204, primals_205, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_8: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_60: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_8, primals_206, primals_207, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_47: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_201: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_47);  sigmoid_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_202: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_201, 2.0);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_203: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, 0.2);  mul_202 = None
    add_51: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, add_46);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_48: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_51)
    mul_204: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_51, sigmoid_48)
    mul_205: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_204, 0.8980265101338745);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_2: "f32[8, 1536, 7, 7]" = torch.ops.aten.avg_pool2d.default(mul_205, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_129: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(primals_130, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_206: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_131, 0.04562504637317021)
    view_130: "f32[1536]" = torch.ops.aten.view.default(mul_206, [-1]);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [0, 2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1536, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1536, 1]" = var_mean_43[1];  var_mean_43 = None
    add_52: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_43: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, getitem_87);  view_129 = None
    mul_207: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_86: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2]);  getitem_87 = None
    squeeze_87: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2]);  rsqrt_43 = None
    unsqueeze_43: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_130, -1);  view_130 = None
    mul_208: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_43);  mul_207 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_131: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_208, [1536, 1536, 1, 1]);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_61: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(avg_pool2d_2, view_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_132: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_133, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_209: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_134, 0.04562504637317021)
    view_133: "f32[384]" = torch.ops.aten.view.default(mul_209, [-1]);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [0, 2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 384, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 384, 1]" = var_mean_44[1];  var_mean_44 = None
    add_53: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_44: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_132, getitem_89);  view_132 = None
    mul_210: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_88: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2]);  getitem_89 = None
    squeeze_89: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2]);  rsqrt_44 = None
    unsqueeze_44: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_133, -1);  view_133 = None
    mul_211: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_44);  mul_210 = unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_134: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_211, [384, 1536, 1, 1]);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_62: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_205, view_134, primals_135, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_49: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_62)
    mul_212: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_62, sigmoid_49);  sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_135: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_136, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_213: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_137, 0.07450538873672485)
    view_136: "f32[384]" = torch.ops.aten.view.default(mul_213, [-1]);  mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [0, 2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 384, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 384, 1]" = var_mean_45[1];  var_mean_45 = None
    add_54: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_45: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_135, getitem_91);  view_135 = None
    mul_214: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_90: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2]);  getitem_91 = None
    squeeze_91: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2]);  rsqrt_45 = None
    unsqueeze_45: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_136, -1);  view_136 = None
    mul_215: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_45);  mul_214 = unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_137: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_215, [384, 64, 3, 3]);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_63: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_212, view_137, primals_138, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_50: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_63)
    mul_216: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_63, sigmoid_50);  sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_138: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_139, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_217: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_140, 0.07450538873672485)
    view_139: "f32[384]" = torch.ops.aten.view.default(mul_217, [-1]);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [0, 2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 384, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 384, 1]" = var_mean_46[1];  var_mean_46 = None
    add_55: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_46: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_138, getitem_93);  view_138 = None
    mul_218: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_92: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2]);  getitem_93 = None
    squeeze_93: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2]);  rsqrt_46 = None
    unsqueeze_46: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_139, -1);  view_139 = None
    mul_219: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_46);  mul_218 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_140: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_219, [384, 64, 3, 3]);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_64: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_216, view_140, primals_141, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_51: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_64)
    mul_220: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, sigmoid_51);  sigmoid_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_141: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_142, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_221: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_143, 0.09125009274634042)
    view_142: "f32[1536]" = torch.ops.aten.view.default(mul_221, [-1]);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [0, 2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 1536, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 1536, 1]" = var_mean_47[1];  var_mean_47 = None
    add_56: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_47: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_141, getitem_95);  view_141 = None
    mul_222: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_94: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2]);  getitem_95 = None
    squeeze_95: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2]);  rsqrt_47 = None
    unsqueeze_47: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_142, -1);  view_142 = None
    mul_223: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_222, unsqueeze_47);  mul_222 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_143: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_223, [1536, 384, 1, 1]);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_65: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(mul_220, view_143, primals_144, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_65, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_66: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_208, primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_9: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_67: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_9, primals_210, primals_211, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_52: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_224: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_52);  sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_225: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_224, 2.0);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_226: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_225, 0.2);  mul_225 = None
    add_57: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_226, convolution_61);  mul_226 = convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_53: "f32[8, 1536, 7, 7]" = torch.ops.aten.sigmoid.default(add_57)
    mul_227: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_57, sigmoid_53)
    mul_228: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_227, 0.9805806756909201);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_144: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_145, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_229: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_146, 0.04562504637317021)
    view_145: "f32[384]" = torch.ops.aten.view.default(mul_229, [-1]);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [0, 2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 384, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 384, 1]" = var_mean_48[1];  var_mean_48 = None
    add_58: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_48: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_144, getitem_97);  view_144 = None
    mul_230: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_96: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2]);  getitem_97 = None
    squeeze_97: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2]);  rsqrt_48 = None
    unsqueeze_48: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_145, -1);  view_145 = None
    mul_231: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_48);  mul_230 = unsqueeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_146: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_231, [384, 1536, 1, 1]);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_68: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_228, view_146, primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_54: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_68)
    mul_232: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, sigmoid_54);  sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_147: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_148, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_233: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_149, 0.07450538873672485)
    view_148: "f32[384]" = torch.ops.aten.view.default(mul_233, [-1]);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [0, 2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 384, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 384, 1]" = var_mean_49[1];  var_mean_49 = None
    add_59: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_49: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_147, getitem_99);  view_147 = None
    mul_234: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_98: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2]);  getitem_99 = None
    squeeze_99: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2]);  rsqrt_49 = None
    unsqueeze_49: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_148, -1);  view_148 = None
    mul_235: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_49);  mul_234 = unsqueeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_149: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_235, [384, 64, 3, 3]);  mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_69: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_232, view_149, primals_150, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_55: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_69)
    mul_236: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_69, sigmoid_55);  sigmoid_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_150: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_151, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_237: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_152, 0.07450538873672485)
    view_151: "f32[384]" = torch.ops.aten.view.default(mul_237, [-1]);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [0, 2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 384, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 384, 1]" = var_mean_50[1];  var_mean_50 = None
    add_60: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_50: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_150, getitem_101);  view_150 = None
    mul_238: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_100: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2]);  getitem_101 = None
    squeeze_101: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2]);  rsqrt_50 = None
    unsqueeze_50: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_151, -1);  view_151 = None
    mul_239: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_50);  mul_238 = unsqueeze_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_152: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_239, [384, 64, 3, 3]);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_70: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_236, view_152, primals_153, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_56: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_70)
    mul_240: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, sigmoid_56);  sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_153: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_154, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_155, 0.09125009274634042)
    view_154: "f32[1536]" = torch.ops.aten.view.default(mul_241, [-1]);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [0, 2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1536, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1536, 1]" = var_mean_51[1];  var_mean_51 = None
    add_61: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_51: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_153, getitem_103);  view_153 = None
    mul_242: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_102: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2]);  getitem_103 = None
    squeeze_103: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2]);  rsqrt_51 = None
    unsqueeze_51: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_154, -1);  view_154 = None
    mul_243: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_51);  mul_242 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_155: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_243, [1536, 384, 1, 1]);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_71: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(mul_240, view_155, primals_156, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_71, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_72: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_212, primals_213, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_10: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_73: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_10, primals_214, primals_215, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_57: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_244: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_57);  sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_245: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_244, 2.0);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_246: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_245, 0.2);  mul_245 = None
    add_62: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_246, add_57);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_58: "f32[8, 1536, 7, 7]" = torch.ops.aten.sigmoid.default(add_62)
    mul_247: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_62, sigmoid_58)
    mul_248: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_247, 0.9622504486493761);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_156: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_157, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_249: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_158, 0.04562504637317021)
    view_157: "f32[384]" = torch.ops.aten.view.default(mul_249, [-1]);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [0, 2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 384, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 384, 1]" = var_mean_52[1];  var_mean_52 = None
    add_63: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_52: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_156, getitem_105);  view_156 = None
    mul_250: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_104: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2]);  getitem_105 = None
    squeeze_105: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2]);  rsqrt_52 = None
    unsqueeze_52: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_157, -1);  view_157 = None
    mul_251: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_52);  mul_250 = unsqueeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_158: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_251, [384, 1536, 1, 1]);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_74: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_248, view_158, primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_59: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_74)
    mul_252: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, sigmoid_59);  sigmoid_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_159: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_160, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_253: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_161, 0.07450538873672485)
    view_160: "f32[384]" = torch.ops.aten.view.default(mul_253, [-1]);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [0, 2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 384, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 384, 1]" = var_mean_53[1];  var_mean_53 = None
    add_64: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_53: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_159, getitem_107);  view_159 = None
    mul_254: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_106: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2]);  getitem_107 = None
    squeeze_107: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2]);  rsqrt_53 = None
    unsqueeze_53: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_160, -1);  view_160 = None
    mul_255: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_53);  mul_254 = unsqueeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_161: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_255, [384, 64, 3, 3]);  mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_75: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_252, view_161, primals_162, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_60: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_75)
    mul_256: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_75, sigmoid_60);  sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_162: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_163, [1, 384, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_257: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_164, 0.07450538873672485)
    view_163: "f32[384]" = torch.ops.aten.view.default(mul_257, [-1]);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [0, 2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 384, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 384, 1]" = var_mean_54[1];  var_mean_54 = None
    add_65: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_54: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = None
    mul_258: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_108: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2]);  getitem_109 = None
    squeeze_109: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2]);  rsqrt_54 = None
    unsqueeze_54: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_163, -1);  view_163 = None
    mul_259: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_54);  mul_258 = unsqueeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_164: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_259, [384, 64, 3, 3]);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_76: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_256, view_164, primals_165, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_61: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_76)
    mul_260: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_76, sigmoid_61);  sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_165: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_166, [1, 1536, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_261: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_167, 0.09125009274634042)
    view_166: "f32[1536]" = torch.ops.aten.view.default(mul_261, [-1]);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [0, 2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1536, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 1536, 1]" = var_mean_55[1];  var_mean_55 = None
    add_66: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_55: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_165, getitem_111);  view_165 = None
    mul_262: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_110: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2]);  getitem_111 = None
    squeeze_111: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2]);  rsqrt_55 = None
    unsqueeze_55: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_166, -1);  view_166 = None
    mul_263: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_55);  mul_262 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_167: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_263, [1536, 384, 1, 1]);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_77: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(mul_260, view_167, primals_168, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_78: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_216, primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_11: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_78);  convolution_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_79: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_218, primals_219, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_62: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_264: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_62);  sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_265: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_264, 2.0);  mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_266: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_265, 0.2);  mul_265 = None
    add_67: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_266, add_62);  mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_168: "f32[1, 2304, 1536]" = torch.ops.aten.view.default(primals_169, [1, 2304, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_267: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_170, 0.04562504637317021)
    view_169: "f32[2304]" = torch.ops.aten.view.default(mul_267, [-1]);  mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [0, 2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 2304, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 2304, 1]" = var_mean_56[1];  var_mean_56 = None
    add_68: "f32[1, 2304, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 2304, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_56: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = None
    mul_268: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_112: "f32[2304]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2]);  getitem_113 = None
    squeeze_113: "f32[2304]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2]);  rsqrt_56 = None
    unsqueeze_56: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(view_169, -1);  view_169 = None
    mul_269: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_56);  mul_268 = unsqueeze_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_170: "f32[2304, 1536, 1, 1]" = torch.ops.aten.view.default(mul_269, [2304, 1536, 1, 1]);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_80: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_67, view_170, primals_171, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:445, code: x = self.final_act(x)
    sigmoid_63: "f32[8, 2304, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_80)
    mul_270: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_80, sigmoid_63);  sigmoid_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_12: "f32[8, 2304, 1, 1]" = torch.ops.aten.mean.dim(mul_270, [-1, -2], True);  mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_171: "f32[8, 2304]" = torch.ops.aten.view.default(mean_12, [8, 2304]);  mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_28: "f32[8, 2304]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[2304, 1000]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_221, clone_28, permute);  primals_221 = None
    permute_1: "f32[1000, 2304]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_57: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(squeeze_112, 0);  squeeze_112 = None
    unsqueeze_58: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_65: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_110, 0);  squeeze_110 = None
    unsqueeze_66: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_73: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_74: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 2);  unsqueeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_81: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_106, 0);  squeeze_106 = None
    unsqueeze_82: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_89: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_104, 0);  squeeze_104 = None
    unsqueeze_90: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    full_default_5: "f32[8, 1536, 7, 7]" = torch.ops.aten.full.default([8, 1536, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_82: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_5, sigmoid_58)
    mul_340: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_62, sub_82);  add_62 = sub_82 = None
    add_74: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Scalar(mul_340, 1);  mul_340 = None
    mul_341: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_74);  sigmoid_58 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_97: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_98: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 2);  unsqueeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_105: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_100, 0);  squeeze_100 = None
    unsqueeze_106: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_113: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_98, 0);  squeeze_98 = None
    unsqueeze_114: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_121: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_122: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sub_103: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_5, sigmoid_53);  full_default_5 = None
    mul_399: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_57, sub_103);  add_57 = sub_103 = None
    add_80: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Scalar(mul_399, 1);  mul_399 = None
    mul_400: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_80);  sigmoid_53 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_129: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_94, 0);  squeeze_94 = None
    unsqueeze_130: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_137: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_92, 0);  squeeze_92 = None
    unsqueeze_138: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_145: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_146: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_153: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_88, 0);  squeeze_88 = None
    unsqueeze_154: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_161: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_86, 0);  squeeze_86 = None
    unsqueeze_162: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    full_default_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.full.default([8, 1536, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_128: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_48)
    mul_468: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_51, sub_128);  add_51 = sub_128 = None
    add_87: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_468, 1);  mul_468 = None
    mul_469: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_48, add_87);  sigmoid_48 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_169: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_170: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_177: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_82, 0);  squeeze_82 = None
    unsqueeze_178: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_185: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_80, 0);  squeeze_80 = None
    unsqueeze_186: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_193: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_194: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sub_149: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_43)
    mul_527: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, sub_149);  add_46 = sub_149 = None
    add_92: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_527, 1);  mul_527 = None
    mul_528: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_92);  sigmoid_43 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_201: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_76, 0);  squeeze_76 = None
    unsqueeze_202: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_209: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_74, 0);  squeeze_74 = None
    unsqueeze_210: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_217: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_218: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_225: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_70, 0);  squeeze_70 = None
    unsqueeze_226: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sub_170: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_38)
    mul_586: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sub_170);  add_41 = sub_170 = None
    add_98: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_586, 1);  mul_586 = None
    mul_587: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_38, add_98);  sigmoid_38 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_233: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_68, 0);  squeeze_68 = None
    unsqueeze_234: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_241: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_242: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_249: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_64, 0);  squeeze_64 = None
    unsqueeze_250: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_257: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_62, 0);  squeeze_62 = None
    unsqueeze_258: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sub_191: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_33)
    mul_645: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_36, sub_191);  add_36 = sub_191 = None
    add_104: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_645, 1);  mul_645 = None
    mul_646: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_33, add_104);  sigmoid_33 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_265: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_266: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_273: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_58, 0);  squeeze_58 = None
    unsqueeze_274: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_281: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_56, 0);  squeeze_56 = None
    unsqueeze_282: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_289: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_290: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sub_212: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_28)
    mul_704: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_31, sub_212);  add_31 = sub_212 = None
    add_110: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_704, 1);  mul_704 = None
    mul_705: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_28, add_110);  sigmoid_28 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_297: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_52, 0);  squeeze_52 = None
    unsqueeze_298: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_305: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_50, 0);  squeeze_50 = None
    unsqueeze_306: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_313: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_314: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_321: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_46, 0);  squeeze_46 = None
    unsqueeze_322: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sub_233: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_23);  full_default_15 = None
    mul_763: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_26, sub_233);  add_26 = sub_233 = None
    add_116: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_763, 1);  mul_763 = None
    mul_764: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_23, add_116);  sigmoid_23 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_329: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_44, 0);  squeeze_44 = None
    unsqueeze_330: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_337: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_338: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_345: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_40, 0);  squeeze_40 = None
    unsqueeze_346: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_353: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_38, 0);  squeeze_38 = None
    unsqueeze_354: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_361: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_362: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    full_default_45: "f32[8, 512, 28, 28]" = torch.ops.aten.full.default([8, 512, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_258: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_45, sigmoid_18)
    mul_832: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_20, sub_258);  add_20 = sub_258 = None
    add_123: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Scalar(mul_832, 1);  mul_832 = None
    mul_833: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_18, add_123);  sigmoid_18 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_369: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_34, 0);  squeeze_34 = None
    unsqueeze_370: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_377: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_32, 0);  squeeze_32 = None
    unsqueeze_378: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_385: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_386: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_393: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_28, 0);  squeeze_28 = None
    unsqueeze_394: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sub_279: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_45, sigmoid_13);  full_default_45 = None
    mul_891: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_15, sub_279);  add_15 = sub_279 = None
    add_128: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Scalar(mul_891, 1);  mul_891 = None
    mul_892: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_13, add_128);  sigmoid_13 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_401: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_26, 0);  squeeze_26 = None
    unsqueeze_402: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_409: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_410: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_417: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_22, 0);  squeeze_22 = None
    unsqueeze_418: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_425: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_20, 0);  squeeze_20 = None
    unsqueeze_426: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_433: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_434: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    full_default_55: "f32[8, 256, 56, 56]" = torch.ops.aten.full.default([8, 256, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_304: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_55, sigmoid_8);  full_default_55 = None
    mul_960: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sub_304);  add_9 = sub_304 = None
    add_135: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Scalar(mul_960, 1);  mul_960 = None
    mul_961: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_8, add_135);  sigmoid_8 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_441: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_16, 0);  squeeze_16 = None
    unsqueeze_442: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_449: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_14, 0);  squeeze_14 = None
    unsqueeze_450: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_457: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_458: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_465: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_10, 0);  squeeze_10 = None
    unsqueeze_466: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_473: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_8, 0);  squeeze_8 = None
    unsqueeze_474: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_481: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_482: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_489: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_4, 0);  squeeze_4 = None
    unsqueeze_490: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_497: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
    unsqueeze_498: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_505: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_506: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    return [addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_222, squeeze_1, view_2, convolution, mul_3, squeeze_3, view_5, convolution_1, mul_7, squeeze_5, view_8, convolution_2, mul_11, squeeze_7, view_11, convolution_3, mul_16, squeeze_9, view_14, squeeze_11, view_17, convolution_5, mul_23, squeeze_13, view_20, convolution_6, mul_27, squeeze_15, view_23, convolution_7, mul_31, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_39, avg_pool2d, squeeze_19, view_29, squeeze_21, view_32, convolution_12, mul_46, squeeze_23, view_35, convolution_13, mul_50, squeeze_25, view_38, convolution_14, mul_54, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_62, squeeze_29, view_44, convolution_18, mul_66, squeeze_31, view_47, convolution_19, mul_70, squeeze_33, view_50, convolution_20, mul_74, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_82, avg_pool2d_1, squeeze_37, view_56, squeeze_39, view_59, convolution_25, mul_89, squeeze_41, view_62, convolution_26, mul_93, squeeze_43, view_65, convolution_27, mul_97, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_105, squeeze_47, view_71, convolution_31, mul_109, squeeze_49, view_74, convolution_32, mul_113, squeeze_51, view_77, convolution_33, mul_117, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_125, squeeze_55, view_83, convolution_37, mul_129, squeeze_57, view_86, convolution_38, mul_133, squeeze_59, view_89, convolution_39, mul_137, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_145, squeeze_63, view_95, convolution_43, mul_149, squeeze_65, view_98, convolution_44, mul_153, squeeze_67, view_101, convolution_45, mul_157, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_165, squeeze_71, view_107, convolution_49, mul_169, squeeze_73, view_110, convolution_50, mul_173, squeeze_75, view_113, convolution_51, mul_177, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_185, squeeze_79, view_119, convolution_55, mul_189, squeeze_81, view_122, convolution_56, mul_193, squeeze_83, view_125, convolution_57, mul_197, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_205, avg_pool2d_2, squeeze_87, view_131, squeeze_89, view_134, convolution_62, mul_212, squeeze_91, view_137, convolution_63, mul_216, squeeze_93, view_140, convolution_64, mul_220, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_228, squeeze_97, view_146, convolution_68, mul_232, squeeze_99, view_149, convolution_69, mul_236, squeeze_101, view_152, convolution_70, mul_240, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_248, squeeze_105, view_158, convolution_74, mul_252, squeeze_107, view_161, convolution_75, mul_256, squeeze_109, view_164, convolution_76, mul_260, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_67, squeeze_113, view_170, convolution_80, clone_28, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, mul_341, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, mul_400, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, mul_469, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, mul_528, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, mul_587, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, mul_646, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, mul_705, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, mul_764, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, mul_833, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, mul_892, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, mul_961, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506]
    