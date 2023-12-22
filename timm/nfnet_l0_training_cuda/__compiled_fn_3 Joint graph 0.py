from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16, 3, 3, 3]"; primals_2: "f32[16, 1, 1, 1]"; primals_3: "f32[16]"; primals_4: "f32[32, 16, 3, 3]"; primals_5: "f32[32, 1, 1, 1]"; primals_6: "f32[32]"; primals_7: "f32[64, 32, 3, 3]"; primals_8: "f32[64, 1, 1, 1]"; primals_9: "f32[64]"; primals_10: "f32[128, 64, 3, 3]"; primals_11: "f32[128, 1, 1, 1]"; primals_12: "f32[128]"; primals_13: "f32[256, 128, 1, 1]"; primals_14: "f32[256, 1, 1, 1]"; primals_15: "f32[256]"; primals_16: "f32[64, 128, 1, 1]"; primals_17: "f32[64, 1, 1, 1]"; primals_18: "f32[64]"; primals_19: "f32[64, 64, 3, 3]"; primals_20: "f32[64, 1, 1, 1]"; primals_21: "f32[64]"; primals_22: "f32[64, 64, 3, 3]"; primals_23: "f32[64, 1, 1, 1]"; primals_24: "f32[64]"; primals_25: "f32[256, 64, 1, 1]"; primals_26: "f32[256, 1, 1, 1]"; primals_27: "f32[256]"; primals_28: "f32[512, 256, 1, 1]"; primals_29: "f32[512, 1, 1, 1]"; primals_30: "f32[512]"; primals_31: "f32[128, 256, 1, 1]"; primals_32: "f32[128, 1, 1, 1]"; primals_33: "f32[128]"; primals_34: "f32[128, 64, 3, 3]"; primals_35: "f32[128, 1, 1, 1]"; primals_36: "f32[128]"; primals_37: "f32[128, 64, 3, 3]"; primals_38: "f32[128, 1, 1, 1]"; primals_39: "f32[128]"; primals_40: "f32[512, 128, 1, 1]"; primals_41: "f32[512, 1, 1, 1]"; primals_42: "f32[512]"; primals_43: "f32[128, 512, 1, 1]"; primals_44: "f32[128, 1, 1, 1]"; primals_45: "f32[128]"; primals_46: "f32[128, 64, 3, 3]"; primals_47: "f32[128, 1, 1, 1]"; primals_48: "f32[128]"; primals_49: "f32[128, 64, 3, 3]"; primals_50: "f32[128, 1, 1, 1]"; primals_51: "f32[128]"; primals_52: "f32[512, 128, 1, 1]"; primals_53: "f32[512, 1, 1, 1]"; primals_54: "f32[512]"; primals_55: "f32[1536, 512, 1, 1]"; primals_56: "f32[1536, 1, 1, 1]"; primals_57: "f32[1536]"; primals_58: "f32[384, 512, 1, 1]"; primals_59: "f32[384, 1, 1, 1]"; primals_60: "f32[384]"; primals_61: "f32[384, 64, 3, 3]"; primals_62: "f32[384, 1, 1, 1]"; primals_63: "f32[384]"; primals_64: "f32[384, 64, 3, 3]"; primals_65: "f32[384, 1, 1, 1]"; primals_66: "f32[384]"; primals_67: "f32[1536, 384, 1, 1]"; primals_68: "f32[1536, 1, 1, 1]"; primals_69: "f32[1536]"; primals_70: "f32[384, 1536, 1, 1]"; primals_71: "f32[384, 1, 1, 1]"; primals_72: "f32[384]"; primals_73: "f32[384, 64, 3, 3]"; primals_74: "f32[384, 1, 1, 1]"; primals_75: "f32[384]"; primals_76: "f32[384, 64, 3, 3]"; primals_77: "f32[384, 1, 1, 1]"; primals_78: "f32[384]"; primals_79: "f32[1536, 384, 1, 1]"; primals_80: "f32[1536, 1, 1, 1]"; primals_81: "f32[1536]"; primals_82: "f32[384, 1536, 1, 1]"; primals_83: "f32[384, 1, 1, 1]"; primals_84: "f32[384]"; primals_85: "f32[384, 64, 3, 3]"; primals_86: "f32[384, 1, 1, 1]"; primals_87: "f32[384]"; primals_88: "f32[384, 64, 3, 3]"; primals_89: "f32[384, 1, 1, 1]"; primals_90: "f32[384]"; primals_91: "f32[1536, 384, 1, 1]"; primals_92: "f32[1536, 1, 1, 1]"; primals_93: "f32[1536]"; primals_94: "f32[384, 1536, 1, 1]"; primals_95: "f32[384, 1, 1, 1]"; primals_96: "f32[384]"; primals_97: "f32[384, 64, 3, 3]"; primals_98: "f32[384, 1, 1, 1]"; primals_99: "f32[384]"; primals_100: "f32[384, 64, 3, 3]"; primals_101: "f32[384, 1, 1, 1]"; primals_102: "f32[384]"; primals_103: "f32[1536, 384, 1, 1]"; primals_104: "f32[1536, 1, 1, 1]"; primals_105: "f32[1536]"; primals_106: "f32[384, 1536, 1, 1]"; primals_107: "f32[384, 1, 1, 1]"; primals_108: "f32[384]"; primals_109: "f32[384, 64, 3, 3]"; primals_110: "f32[384, 1, 1, 1]"; primals_111: "f32[384]"; primals_112: "f32[384, 64, 3, 3]"; primals_113: "f32[384, 1, 1, 1]"; primals_114: "f32[384]"; primals_115: "f32[1536, 384, 1, 1]"; primals_116: "f32[1536, 1, 1, 1]"; primals_117: "f32[1536]"; primals_118: "f32[384, 1536, 1, 1]"; primals_119: "f32[384, 1, 1, 1]"; primals_120: "f32[384]"; primals_121: "f32[384, 64, 3, 3]"; primals_122: "f32[384, 1, 1, 1]"; primals_123: "f32[384]"; primals_124: "f32[384, 64, 3, 3]"; primals_125: "f32[384, 1, 1, 1]"; primals_126: "f32[384]"; primals_127: "f32[1536, 384, 1, 1]"; primals_128: "f32[1536, 1, 1, 1]"; primals_129: "f32[1536]"; primals_130: "f32[1536, 1536, 1, 1]"; primals_131: "f32[1536, 1, 1, 1]"; primals_132: "f32[1536]"; primals_133: "f32[384, 1536, 1, 1]"; primals_134: "f32[384, 1, 1, 1]"; primals_135: "f32[384]"; primals_136: "f32[384, 64, 3, 3]"; primals_137: "f32[384, 1, 1, 1]"; primals_138: "f32[384]"; primals_139: "f32[384, 64, 3, 3]"; primals_140: "f32[384, 1, 1, 1]"; primals_141: "f32[384]"; primals_142: "f32[1536, 384, 1, 1]"; primals_143: "f32[1536, 1, 1, 1]"; primals_144: "f32[1536]"; primals_145: "f32[384, 1536, 1, 1]"; primals_146: "f32[384, 1, 1, 1]"; primals_147: "f32[384]"; primals_148: "f32[384, 64, 3, 3]"; primals_149: "f32[384, 1, 1, 1]"; primals_150: "f32[384]"; primals_151: "f32[384, 64, 3, 3]"; primals_152: "f32[384, 1, 1, 1]"; primals_153: "f32[384]"; primals_154: "f32[1536, 384, 1, 1]"; primals_155: "f32[1536, 1, 1, 1]"; primals_156: "f32[1536]"; primals_157: "f32[384, 1536, 1, 1]"; primals_158: "f32[384, 1, 1, 1]"; primals_159: "f32[384]"; primals_160: "f32[384, 64, 3, 3]"; primals_161: "f32[384, 1, 1, 1]"; primals_162: "f32[384]"; primals_163: "f32[384, 64, 3, 3]"; primals_164: "f32[384, 1, 1, 1]"; primals_165: "f32[384]"; primals_166: "f32[1536, 384, 1, 1]"; primals_167: "f32[1536, 1, 1, 1]"; primals_168: "f32[1536]"; primals_169: "f32[2304, 1536, 1, 1]"; primals_170: "f32[2304, 1, 1, 1]"; primals_171: "f32[2304]"; primals_172: "f32[64, 256, 1, 1]"; primals_173: "f32[64]"; primals_174: "f32[256, 64, 1, 1]"; primals_175: "f32[256]"; primals_176: "f32[128, 512, 1, 1]"; primals_177: "f32[128]"; primals_178: "f32[512, 128, 1, 1]"; primals_179: "f32[512]"; primals_180: "f32[128, 512, 1, 1]"; primals_181: "f32[128]"; primals_182: "f32[512, 128, 1, 1]"; primals_183: "f32[512]"; primals_184: "f32[384, 1536, 1, 1]"; primals_185: "f32[384]"; primals_186: "f32[1536, 384, 1, 1]"; primals_187: "f32[1536]"; primals_188: "f32[384, 1536, 1, 1]"; primals_189: "f32[384]"; primals_190: "f32[1536, 384, 1, 1]"; primals_191: "f32[1536]"; primals_192: "f32[384, 1536, 1, 1]"; primals_193: "f32[384]"; primals_194: "f32[1536, 384, 1, 1]"; primals_195: "f32[1536]"; primals_196: "f32[384, 1536, 1, 1]"; primals_197: "f32[384]"; primals_198: "f32[1536, 384, 1, 1]"; primals_199: "f32[1536]"; primals_200: "f32[384, 1536, 1, 1]"; primals_201: "f32[384]"; primals_202: "f32[1536, 384, 1, 1]"; primals_203: "f32[1536]"; primals_204: "f32[384, 1536, 1, 1]"; primals_205: "f32[384]"; primals_206: "f32[1536, 384, 1, 1]"; primals_207: "f32[1536]"; primals_208: "f32[384, 1536, 1, 1]"; primals_209: "f32[384]"; primals_210: "f32[1536, 384, 1, 1]"; primals_211: "f32[1536]"; primals_212: "f32[384, 1536, 1, 1]"; primals_213: "f32[384]"; primals_214: "f32[1536, 384, 1, 1]"; primals_215: "f32[1536]"; primals_216: "f32[384, 1536, 1, 1]"; primals_217: "f32[384]"; primals_218: "f32[1536, 384, 1, 1]"; primals_219: "f32[1536]"; primals_220: "f32[1000, 2304]"; primals_221: "f32[1000]"; primals_222: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view: "f32[1, 16, 27]" = torch.ops.aten.view.default(primals_1, [1, 16, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_2, 0.34412564994580647);  primals_2 = None
    view_1: "f32[16]" = torch.ops.aten.view.default(mul, [-1]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(view, [0, 2], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, getitem_1)
    mul_1: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2]);  rsqrt = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    mul_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze);  mul_1 = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_2: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_2, [16, 3, 3, 3]);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_222, view_2, primals_3, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    clone: "f32[8, 16, 112, 112]" = torch.ops.aten.clone.default(convolution)
    sigmoid: "f32[8, 16, 112, 112]" = torch.ops.aten.sigmoid.default(convolution)
    mul_3: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(convolution, sigmoid);  convolution = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_3: "f32[1, 32, 144]" = torch.ops.aten.view.default(primals_4, [1, 32, -1]);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_4: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_5, 0.1490107774734497);  primals_5 = None
    view_4: "f32[32]" = torch.ops.aten.view.default(mul_4, [-1]);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [0, 2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1]" = var_mean_1[1];  var_mean_1 = None
    add_1: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub_1: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, getitem_3)
    mul_5: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2]);  getitem_3 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2]);  rsqrt_1 = None
    unsqueeze_1: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_4, -1)
    mul_6: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_1);  mul_5 = unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_5: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_6, [32, 16, 3, 3]);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_3, view_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    clone_1: "f32[8, 32, 112, 112]" = torch.ops.aten.clone.default(convolution_1)
    sigmoid_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(convolution_1)
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(convolution_1, sigmoid_1);  convolution_1 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_6: "f32[1, 64, 288]" = torch.ops.aten.view.default(primals_7, [1, 64, -1]);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_8: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_8, 0.10536653122135592);  primals_8 = None
    view_7: "f32[64]" = torch.ops.aten.view.default(mul_8, [-1]);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [0, 2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None
    add_2: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_2: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, getitem_5)
    mul_9: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2]);  getitem_5 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2]);  rsqrt_2 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_7, -1)
    mul_10: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_2);  mul_9 = unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_8: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_10, [64, 32, 3, 3]);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_2: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(mul_7, view_8, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    clone_2: "f32[8, 64, 112, 112]" = torch.ops.aten.clone.default(convolution_2)
    sigmoid_2: "f32[8, 64, 112, 112]" = torch.ops.aten.sigmoid.default(convolution_2)
    mul_11: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(convolution_2, sigmoid_2);  convolution_2 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_9: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_10, [1, 128, -1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_12: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_11, 0.07450538873672485);  primals_11 = None
    view_10: "f32[128]" = torch.ops.aten.view.default(mul_12, [-1]);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [0, 2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_3: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, getitem_7)
    mul_13: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2]);  getitem_7 = None
    squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2]);  rsqrt_3 = None
    unsqueeze_3: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_10, -1)
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
    view_12: "f32[1, 256, 128]" = torch.ops.aten.view.default(primals_13, [1, 256, -1]);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_17: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_14, 0.1580497968320339);  primals_14 = None
    view_13: "f32[256]" = torch.ops.aten.view.default(mul_17, [-1]);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [0, 2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 256, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 256, 1]" = var_mean_4[1];  var_mean_4 = None
    add_4: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_4: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, getitem_9)
    mul_18: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_8: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2]);  getitem_9 = None
    squeeze_9: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2]);  rsqrt_4 = None
    unsqueeze_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_13, -1)
    mul_19: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_4);  mul_18 = unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_14: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_19, [256, 128, 1, 1]);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_4: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(mul_16, view_14, primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_15: "f32[1, 64, 128]" = torch.ops.aten.view.default(primals_16, [1, 64, -1]);  primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_20: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_17, 0.1580497968320339);  primals_17 = None
    view_16: "f32[64]" = torch.ops.aten.view.default(mul_20, [-1]);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0, 2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 64, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 64, 1]" = var_mean_5[1];  var_mean_5 = None
    add_5: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_5: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_11)
    mul_21: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2]);  getitem_11 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2]);  rsqrt_5 = None
    unsqueeze_5: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_16, -1)
    mul_22: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_5);  mul_21 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_17: "f32[64, 128, 1, 1]" = torch.ops.aten.view.default(mul_22, [64, 128, 1, 1]);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(mul_16, view_17, primals_18, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_3: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(convolution_5)
    sigmoid_4: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_5)
    mul_23: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, sigmoid_4);  convolution_5 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_18: "f32[1, 64, 576]" = torch.ops.aten.view.default(primals_19, [1, 64, -1]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_24: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_20, 0.07450538873672485);  primals_20 = None
    view_19: "f32[64]" = torch.ops.aten.view.default(mul_24, [-1]);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [0, 2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 64, 1]" = var_mean_6[1];  var_mean_6 = None
    add_6: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_6: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_18, getitem_13)
    mul_25: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2]);  getitem_13 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2]);  rsqrt_6 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_19, -1)
    mul_26: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_6);  mul_25 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_20: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_26, [64, 64, 3, 3]);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_6: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(mul_23, view_20, primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_4: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(convolution_6)
    sigmoid_5: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_6)
    mul_27: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_6, sigmoid_5);  convolution_6 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_21: "f32[1, 64, 576]" = torch.ops.aten.view.default(primals_22, [1, 64, -1]);  primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_28: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_23, 0.07450538873672485);  primals_23 = None
    view_22: "f32[64]" = torch.ops.aten.view.default(mul_28, [-1]);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [0, 2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 64, 1]" = var_mean_7[1];  var_mean_7 = None
    add_7: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_7: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_21, getitem_15)
    mul_29: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2]);  getitem_15 = None
    squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2]);  rsqrt_7 = None
    unsqueeze_7: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_22, -1)
    mul_30: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_7);  mul_29 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_23: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_30, [64, 64, 3, 3]);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_7: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(mul_27, view_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_6: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_7)
    mul_31: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_24: "f32[1, 256, 64]" = torch.ops.aten.view.default(primals_25, [1, 256, -1]);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_32: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_26, 0.22351616621017456);  primals_26 = None
    view_25: "f32[256]" = torch.ops.aten.view.default(mul_32, [-1]);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [0, 2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 256, 1]" = var_mean_8[1];  var_mean_8 = None
    add_8: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_8: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_24, getitem_17)
    mul_33: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2]);  getitem_17 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2]);  rsqrt_8 = None
    unsqueeze_8: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_25, -1)
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
    sigmoid_7: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    alias_1: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_35: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_36: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, 2.0);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_37: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_36, 0.2);  mul_36 = None
    add_9: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_37, convolution_4);  mul_37 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_8: "f32[8, 256, 56, 56]" = torch.ops.aten.sigmoid.default(add_9)
    mul_38: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_8);  sigmoid_8 = None
    mul_39: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_38, 0.9805806756909201);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d: "f32[8, 256, 28, 28]" = torch.ops.aten.avg_pool2d.default(mul_39, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_28, [1, 512, -1]);  primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_40: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_29, 0.11175808310508728);  primals_29 = None
    view_28: "f32[512]" = torch.ops.aten.view.default(mul_40, [-1]);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [0, 2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_19)
    mul_41: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_18: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2]);  getitem_19 = None
    squeeze_19: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2]);  rsqrt_9 = None
    unsqueeze_9: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_28, -1)
    mul_42: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_9);  mul_41 = unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_29: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_42, [512, 256, 1, 1]);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_11: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(avg_pool2d, view_29, primals_30, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_30: "f32[1, 128, 256]" = torch.ops.aten.view.default(primals_31, [1, 128, -1]);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_43: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_32, 0.11175808310508728);  primals_32 = None
    view_31: "f32[128]" = torch.ops.aten.view.default(mul_43, [-1]);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [0, 2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_10: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_30, getitem_21)
    mul_44: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_20: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2]);  getitem_21 = None
    squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2]);  rsqrt_10 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_31, -1)
    mul_45: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_10);  mul_44 = unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_32: "f32[128, 256, 1, 1]" = torch.ops.aten.view.default(mul_45, [128, 256, 1, 1]);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_12: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(mul_39, view_32, primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_5: "f32[8, 128, 56, 56]" = torch.ops.aten.clone.default(convolution_12)
    sigmoid_9: "f32[8, 128, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_12)
    mul_46: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_12, sigmoid_9);  convolution_12 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_33: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_34, [1, 128, -1]);  primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_47: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_35, 0.07450538873672485);  primals_35 = None
    view_34: "f32[128]" = torch.ops.aten.view.default(mul_47, [-1]);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [0, 2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_11: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_33, getitem_23)
    mul_48: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2]);  getitem_23 = None
    squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2]);  rsqrt_11 = None
    unsqueeze_11: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_34, -1)
    mul_49: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_11);  mul_48 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_35: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_49, [128, 64, 3, 3]);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_13: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_46, view_35, primals_36, [2, 2], [1, 1], [1, 1], False, [0, 0], 2);  primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_6: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(convolution_13)
    sigmoid_10: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_13)
    mul_50: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_13, sigmoid_10);  convolution_13 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_36: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_37, [1, 128, -1]);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_51: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_38, 0.07450538873672485);  primals_38 = None
    view_37: "f32[128]" = torch.ops.aten.view.default(mul_51, [-1]);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [0, 2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_13: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_12: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_36, getitem_25)
    mul_52: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_24: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2]);  getitem_25 = None
    squeeze_25: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2]);  rsqrt_12 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_37, -1)
    mul_53: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_12);  mul_52 = unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_38: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_53, [128, 64, 3, 3]);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_14: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_50, view_38, primals_39, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_11: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_14)
    mul_54: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, sigmoid_11);  sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_39: "f32[1, 512, 128]" = torch.ops.aten.view.default(primals_40, [1, 512, -1]);  primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_55: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_41, 0.1580497968320339);  primals_41 = None
    view_40: "f32[512]" = torch.ops.aten.view.default(mul_55, [-1]);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [0, 2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_13: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_39, getitem_27)
    mul_56: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_26: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2]);  getitem_27 = None
    squeeze_27: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2]);  rsqrt_13 = None
    unsqueeze_13: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_40, -1)
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
    sigmoid_12: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    alias_3: "f32[8, 512, 1, 1]" = torch.ops.aten.alias.default(sigmoid_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_58: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_59: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, 2.0);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_60: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, 0.2);  mul_59 = None
    add_15: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_60, convolution_11);  mul_60 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_13: "f32[8, 512, 28, 28]" = torch.ops.aten.sigmoid.default(add_15)
    mul_61: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_15, sigmoid_13);  sigmoid_13 = None
    mul_62: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, 0.9805806756909201);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_42: "f32[1, 128, 512]" = torch.ops.aten.view.default(primals_43, [1, 128, -1]);  primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_63: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_44, 0.07902489841601695);  primals_44 = None
    view_43: "f32[128]" = torch.ops.aten.view.default(mul_63, [-1]);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [0, 2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_16: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_14: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_42, getitem_29)
    mul_64: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_28: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2]);  getitem_29 = None
    squeeze_29: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2]);  rsqrt_14 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_43, -1)
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_14);  mul_64 = unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_44: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(mul_65, [128, 512, 1, 1]);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_18: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_62, view_44, primals_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_7: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(convolution_18)
    sigmoid_14: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_18)
    mul_66: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, sigmoid_14);  convolution_18 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_45: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_46, [1, 128, -1]);  primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_67: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_47, 0.07450538873672485);  primals_47 = None
    view_46: "f32[128]" = torch.ops.aten.view.default(mul_67, [-1]);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [0, 2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_15: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_45, getitem_31)
    mul_68: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2]);  getitem_31 = None
    squeeze_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2]);  rsqrt_15 = None
    unsqueeze_15: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_46, -1)
    mul_69: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_15);  mul_68 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_47: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_69, [128, 64, 3, 3]);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_19: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_66, view_47, primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_8: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(convolution_19)
    sigmoid_15: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_19)
    mul_70: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_19, sigmoid_15);  convolution_19 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_48: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_49, [1, 128, -1]);  primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_71: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_50, 0.07450538873672485);  primals_50 = None
    view_49: "f32[128]" = torch.ops.aten.view.default(mul_71, [-1]);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [0, 2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_16: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_48, getitem_33)
    mul_72: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2]);  getitem_33 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2]);  rsqrt_16 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_49, -1)
    mul_73: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_16);  mul_72 = unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_50: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_73, [128, 64, 3, 3]);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_20: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_70, view_50, primals_51, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_16: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_20)
    mul_74: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_51: "f32[1, 512, 128]" = torch.ops.aten.view.default(primals_52, [1, 512, -1]);  primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_75: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_53, 0.1580497968320339);  primals_53 = None
    view_52: "f32[512]" = torch.ops.aten.view.default(mul_75, [-1]);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [0, 2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_19: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_17: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_51, getitem_35)
    mul_76: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_34: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2]);  getitem_35 = None
    squeeze_35: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2]);  rsqrt_17 = None
    unsqueeze_17: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_52, -1)
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
    sigmoid_17: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_5: "f32[8, 512, 1, 1]" = torch.ops.aten.alias.default(sigmoid_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_78: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_79: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_78, 2.0);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_80: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, 0.2);  mul_79 = None
    add_20: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_80, add_15);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_18: "f32[8, 512, 28, 28]" = torch.ops.aten.sigmoid.default(add_20)
    mul_81: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_20, sigmoid_18);  sigmoid_18 = None
    mul_82: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_81, 0.9622504486493761);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_1: "f32[8, 512, 14, 14]" = torch.ops.aten.avg_pool2d.default(mul_82, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_54: "f32[1, 1536, 512]" = torch.ops.aten.view.default(primals_55, [1, 1536, -1]);  primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_83: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_56, 0.07902489841601695);  primals_56 = None
    view_55: "f32[1536]" = torch.ops.aten.view.default(mul_83, [-1]);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [0, 2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1536, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1536, 1]" = var_mean_18[1];  var_mean_18 = None
    add_21: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_18: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, getitem_37)
    mul_84: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_36: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2]);  getitem_37 = None
    squeeze_37: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2]);  rsqrt_18 = None
    unsqueeze_18: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_55, -1)
    mul_85: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_18);  mul_84 = unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_56: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_85, [1536, 512, 1, 1]);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(avg_pool2d_1, view_56, primals_57, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_57: "f32[1, 384, 512]" = torch.ops.aten.view.default(primals_58, [1, 384, -1]);  primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_86: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_59, 0.07902489841601695);  primals_59 = None
    view_58: "f32[384]" = torch.ops.aten.view.default(mul_86, [-1]);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [0, 2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 384, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 384, 1]" = var_mean_19[1];  var_mean_19 = None
    add_22: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_19: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_57, getitem_39)
    mul_87: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_38: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2]);  getitem_39 = None
    squeeze_39: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2]);  rsqrt_19 = None
    unsqueeze_19: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_58, -1)
    mul_88: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_19);  mul_87 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_59: "f32[384, 512, 1, 1]" = torch.ops.aten.view.default(mul_88, [384, 512, 1, 1]);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_25: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_82, view_59, primals_60, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_9: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(convolution_25)
    sigmoid_19: "f32[8, 384, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_25)
    mul_89: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_25, sigmoid_19);  convolution_25 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_60: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_61, [1, 384, -1]);  primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_90: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_62, 0.07450538873672485);  primals_62 = None
    view_61: "f32[384]" = torch.ops.aten.view.default(mul_90, [-1]);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [0, 2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 384, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 384, 1]" = var_mean_20[1];  var_mean_20 = None
    add_23: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_20: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_60, getitem_41)
    mul_91: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_40: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2]);  getitem_41 = None
    squeeze_41: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2]);  rsqrt_20 = None
    unsqueeze_20: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_61, -1)
    mul_92: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_20);  mul_91 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_62: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_92, [384, 64, 3, 3]);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_26: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_89, view_62, primals_63, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_10: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_26)
    sigmoid_20: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_26)
    mul_93: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, sigmoid_20);  convolution_26 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_63: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_64, [1, 384, -1]);  primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_94: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_65, 0.07450538873672485);  primals_65 = None
    view_64: "f32[384]" = torch.ops.aten.view.default(mul_94, [-1]);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [0, 2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 384, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 384, 1]" = var_mean_21[1];  var_mean_21 = None
    add_24: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_21: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_63, getitem_43)
    mul_95: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_42: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2]);  getitem_43 = None
    squeeze_43: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2]);  rsqrt_21 = None
    unsqueeze_21: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_64, -1)
    mul_96: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_21);  mul_95 = unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_65: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_96, [384, 64, 3, 3]);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_27: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_93, view_65, primals_66, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_21: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_97: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_66: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_67, [1, 1536, -1]);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_98: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_68, 0.09125009274634042);  primals_68 = None
    view_67: "f32[1536]" = torch.ops.aten.view.default(mul_98, [-1]);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [0, 2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1536, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1536, 1]" = var_mean_22[1];  var_mean_22 = None
    add_25: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_22: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_66, getitem_45)
    mul_99: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_44: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2]);  getitem_45 = None
    squeeze_45: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2]);  rsqrt_22 = None
    unsqueeze_22: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_67, -1)
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
    sigmoid_22: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None
    alias_7: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_101: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_102: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_101, 2.0);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_103: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_102, 0.2);  mul_102 = None
    add_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_103, convolution_24);  mul_103 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_23: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_26)
    mul_104: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_26, sigmoid_23);  sigmoid_23 = None
    mul_105: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_104, 0.9805806756909201);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_69: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_70, [1, 384, -1]);  primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_106: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_71, 0.04562504637317021);  primals_71 = None
    view_70: "f32[384]" = torch.ops.aten.view.default(mul_106, [-1]);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [0, 2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 384, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 384, 1]" = var_mean_23[1];  var_mean_23 = None
    add_27: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_23: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_69, getitem_47)
    mul_107: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_46: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2]);  getitem_47 = None
    squeeze_47: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2]);  rsqrt_23 = None
    unsqueeze_23: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_70, -1)
    mul_108: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_23);  mul_107 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_71: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_108, [384, 1536, 1, 1]);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_31: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_105, view_71, primals_72, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_11: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_31)
    sigmoid_24: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_31)
    mul_109: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, sigmoid_24);  convolution_31 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_72: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_73, [1, 384, -1]);  primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_110: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_74, 0.07450538873672485);  primals_74 = None
    view_73: "f32[384]" = torch.ops.aten.view.default(mul_110, [-1]);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [0, 2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 384, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 384, 1]" = var_mean_24[1];  var_mean_24 = None
    add_28: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_24: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_72, getitem_49)
    mul_111: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_48: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2]);  getitem_49 = None
    squeeze_49: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2]);  rsqrt_24 = None
    unsqueeze_24: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_73, -1)
    mul_112: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_24);  mul_111 = unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_74: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_112, [384, 64, 3, 3]);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_32: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_109, view_74, primals_75, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_12: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_32)
    sigmoid_25: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_32)
    mul_113: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_32, sigmoid_25);  convolution_32 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_75: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_76, [1, 384, -1]);  primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_114: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_77, 0.07450538873672485);  primals_77 = None
    view_76: "f32[384]" = torch.ops.aten.view.default(mul_114, [-1]);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [0, 2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 384, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 384, 1]" = var_mean_25[1];  var_mean_25 = None
    add_29: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_25: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_75, getitem_51)
    mul_115: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_50: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2]);  getitem_51 = None
    squeeze_51: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2]);  rsqrt_25 = None
    unsqueeze_25: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_76, -1)
    mul_116: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_25);  mul_115 = unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_77: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_116, [384, 64, 3, 3]);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_33: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_113, view_77, primals_78, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_26: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_33)
    mul_117: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_78: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_79, [1, 1536, -1]);  primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_118: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_80, 0.09125009274634042);  primals_80 = None
    view_79: "f32[1536]" = torch.ops.aten.view.default(mul_118, [-1]);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [0, 2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 1536, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 1536, 1]" = var_mean_26[1];  var_mean_26 = None
    add_30: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_26: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_78, getitem_53)
    mul_119: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_52: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2]);  getitem_53 = None
    squeeze_53: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2]);  rsqrt_26 = None
    unsqueeze_26: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_79, -1)
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
    sigmoid_27: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    alias_9: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_121: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_122: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, 2.0);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_123: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_122, 0.2);  mul_122 = None
    add_31: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_123, add_26);  mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_31)
    mul_124: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_31, sigmoid_28);  sigmoid_28 = None
    mul_125: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, 0.9622504486493761);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_81: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_82, [1, 384, -1]);  primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_126: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_83, 0.04562504637317021);  primals_83 = None
    view_82: "f32[384]" = torch.ops.aten.view.default(mul_126, [-1]);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [0, 2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 384, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 384, 1]" = var_mean_27[1];  var_mean_27 = None
    add_32: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_27: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_81, getitem_55)
    mul_127: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_54: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2]);  getitem_55 = None
    squeeze_55: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2]);  rsqrt_27 = None
    unsqueeze_27: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_82, -1)
    mul_128: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_27);  mul_127 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_83: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_128, [384, 1536, 1, 1]);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_37: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_125, view_83, primals_84, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_13: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_37)
    sigmoid_29: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_37)
    mul_129: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, sigmoid_29);  convolution_37 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_84: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_85, [1, 384, -1]);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_130: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_86, 0.07450538873672485);  primals_86 = None
    view_85: "f32[384]" = torch.ops.aten.view.default(mul_130, [-1]);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [0, 2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 384, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 384, 1]" = var_mean_28[1];  var_mean_28 = None
    add_33: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_28: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_84, getitem_57)
    mul_131: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_56: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2]);  getitem_57 = None
    squeeze_57: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2]);  rsqrt_28 = None
    unsqueeze_28: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_85, -1)
    mul_132: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_28);  mul_131 = unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_86: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_132, [384, 64, 3, 3]);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_38: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_129, view_86, primals_87, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_14: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_38)
    sigmoid_30: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_38)
    mul_133: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, sigmoid_30);  convolution_38 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_87: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_88, [1, 384, -1]);  primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_134: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_89, 0.07450538873672485);  primals_89 = None
    view_88: "f32[384]" = torch.ops.aten.view.default(mul_134, [-1]);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [0, 2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 384, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 384, 1]" = var_mean_29[1];  var_mean_29 = None
    add_34: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_29: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_87, getitem_59)
    mul_135: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_58: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2]);  getitem_59 = None
    squeeze_59: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2]);  rsqrt_29 = None
    unsqueeze_29: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_88, -1)
    mul_136: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_29);  mul_135 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_89: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_136, [384, 64, 3, 3]);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_39: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_133, view_89, primals_90, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_31: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_39)
    mul_137: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, sigmoid_31);  sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_90: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_91, [1, 1536, -1]);  primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_138: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_92, 0.09125009274634042);  primals_92 = None
    view_91: "f32[1536]" = torch.ops.aten.view.default(mul_138, [-1]);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [0, 2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 1536, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 1536, 1]" = var_mean_30[1];  var_mean_30 = None
    add_35: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_30: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_90, getitem_61)
    mul_139: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_60: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2]);  getitem_61 = None
    squeeze_61: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2]);  rsqrt_30 = None
    unsqueeze_30: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_91, -1)
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
    sigmoid_32: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None
    alias_11: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_141: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_142: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_141, 2.0);  mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_143: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_142, 0.2);  mul_142 = None
    add_36: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, add_31);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_33: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_36)
    mul_144: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_36, sigmoid_33);  sigmoid_33 = None
    mul_145: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_144, 0.9449111825230679);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_93: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_94, [1, 384, -1]);  primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_146: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_95, 0.04562504637317021);  primals_95 = None
    view_94: "f32[384]" = torch.ops.aten.view.default(mul_146, [-1]);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [0, 2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 384, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 384, 1]" = var_mean_31[1];  var_mean_31 = None
    add_37: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_31: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_93, getitem_63)
    mul_147: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_62: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2]);  getitem_63 = None
    squeeze_63: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2]);  rsqrt_31 = None
    unsqueeze_31: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_94, -1)
    mul_148: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_31);  mul_147 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_95: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_148, [384, 1536, 1, 1]);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_43: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_145, view_95, primals_96, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_15: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_43)
    sigmoid_34: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_149: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, sigmoid_34);  convolution_43 = sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_96: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_97, [1, 384, -1]);  primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_150: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_98, 0.07450538873672485);  primals_98 = None
    view_97: "f32[384]" = torch.ops.aten.view.default(mul_150, [-1]);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [0, 2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 384, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 384, 1]" = var_mean_32[1];  var_mean_32 = None
    add_38: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_32: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_96, getitem_65)
    mul_151: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_64: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2]);  getitem_65 = None
    squeeze_65: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2]);  rsqrt_32 = None
    unsqueeze_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_97, -1)
    mul_152: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_32);  mul_151 = unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_98: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_152, [384, 64, 3, 3]);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_44: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_149, view_98, primals_99, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_16: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_44)
    sigmoid_35: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_44)
    mul_153: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_44, sigmoid_35);  convolution_44 = sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_99: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_100, [1, 384, -1]);  primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_154: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_101, 0.07450538873672485);  primals_101 = None
    view_100: "f32[384]" = torch.ops.aten.view.default(mul_154, [-1]);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [0, 2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 384, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 384, 1]" = var_mean_33[1];  var_mean_33 = None
    add_39: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_33: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_99, getitem_67)
    mul_155: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_66: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2]);  getitem_67 = None
    squeeze_67: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2]);  rsqrt_33 = None
    unsqueeze_33: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_100, -1)
    mul_156: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_33);  mul_155 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_101: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_156, [384, 64, 3, 3]);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_45: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_153, view_101, primals_102, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_36: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_45)
    mul_157: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, sigmoid_36);  sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_102: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_103, [1, 1536, -1]);  primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_158: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_104, 0.09125009274634042);  primals_104 = None
    view_103: "f32[1536]" = torch.ops.aten.view.default(mul_158, [-1]);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [0, 2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1536, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 1536, 1]" = var_mean_34[1];  var_mean_34 = None
    add_40: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_34: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_102, getitem_69)
    mul_159: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_68: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2]);  getitem_69 = None
    squeeze_69: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2]);  rsqrt_34 = None
    unsqueeze_34: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_103, -1)
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
    sigmoid_37: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    alias_13: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_161: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_162: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_161, 2.0);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_163: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_162, 0.2);  mul_162 = None
    add_41: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_163, add_36);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_41)
    mul_164: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_38);  sigmoid_38 = None
    mul_165: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_164, 0.9284766908852592);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_105: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_106, [1, 384, -1]);  primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_166: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_107, 0.04562504637317021);  primals_107 = None
    view_106: "f32[384]" = torch.ops.aten.view.default(mul_166, [-1]);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [0, 2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 384, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 384, 1]" = var_mean_35[1];  var_mean_35 = None
    add_42: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_35: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_105, getitem_71)
    mul_167: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_70: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2]);  getitem_71 = None
    squeeze_71: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2]);  rsqrt_35 = None
    unsqueeze_35: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_106, -1)
    mul_168: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_35);  mul_167 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_107: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_168, [384, 1536, 1, 1]);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_49: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_165, view_107, primals_108, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_17: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_49)
    sigmoid_39: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_49)
    mul_169: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, sigmoid_39);  convolution_49 = sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_108: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_109, [1, 384, -1]);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_170: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_110, 0.07450538873672485);  primals_110 = None
    view_109: "f32[384]" = torch.ops.aten.view.default(mul_170, [-1]);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [0, 2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 384, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 384, 1]" = var_mean_36[1];  var_mean_36 = None
    add_43: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_36: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_108, getitem_73)
    mul_171: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_72: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2]);  getitem_73 = None
    squeeze_73: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2]);  rsqrt_36 = None
    unsqueeze_36: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_109, -1)
    mul_172: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_36);  mul_171 = unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_110: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_172, [384, 64, 3, 3]);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_50: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_169, view_110, primals_111, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_18: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_50)
    sigmoid_40: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_50)
    mul_173: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_50, sigmoid_40);  convolution_50 = sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_111: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_112, [1, 384, -1]);  primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_174: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_113, 0.07450538873672485);  primals_113 = None
    view_112: "f32[384]" = torch.ops.aten.view.default(mul_174, [-1]);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [0, 2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 384, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 384, 1]" = var_mean_37[1];  var_mean_37 = None
    add_44: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_37: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_111, getitem_75)
    mul_175: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_74: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2]);  getitem_75 = None
    squeeze_75: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2]);  rsqrt_37 = None
    unsqueeze_37: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_112, -1)
    mul_176: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_37);  mul_175 = unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_113: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_176, [384, 64, 3, 3]);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_51: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_173, view_113, primals_114, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_41: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_51)
    mul_177: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, sigmoid_41);  sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_114: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_115, [1, 1536, -1]);  primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_116, 0.09125009274634042);  primals_116 = None
    view_115: "f32[1536]" = torch.ops.aten.view.default(mul_178, [-1]);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [0, 2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1536, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 1536, 1]" = var_mean_38[1];  var_mean_38 = None
    add_45: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_38: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_114, getitem_77)
    mul_179: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_76: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2]);  getitem_77 = None
    squeeze_77: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2]);  rsqrt_38 = None
    unsqueeze_38: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_115, -1)
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
    sigmoid_42: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None
    alias_15: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_181: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_182: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, 2.0);  mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_183: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, 0.2);  mul_182 = None
    add_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_183, add_41);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_43: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_46)
    mul_184: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, sigmoid_43);  sigmoid_43 = None
    mul_185: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, 0.9128709291752768);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_117: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_118, [1, 384, -1]);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_186: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_119, 0.04562504637317021);  primals_119 = None
    view_118: "f32[384]" = torch.ops.aten.view.default(mul_186, [-1]);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [0, 2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 384, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 384, 1]" = var_mean_39[1];  var_mean_39 = None
    add_47: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_39: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_117, getitem_79)
    mul_187: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_78: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2]);  getitem_79 = None
    squeeze_79: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2]);  rsqrt_39 = None
    unsqueeze_39: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_118, -1)
    mul_188: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_39);  mul_187 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_119: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_188, [384, 1536, 1, 1]);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_55: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_185, view_119, primals_120, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_19: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_55)
    sigmoid_44: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_55)
    mul_189: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, sigmoid_44);  convolution_55 = sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_120: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_121, [1, 384, -1]);  primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_190: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_122, 0.07450538873672485);  primals_122 = None
    view_121: "f32[384]" = torch.ops.aten.view.default(mul_190, [-1]);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [0, 2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 384, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 384, 1]" = var_mean_40[1];  var_mean_40 = None
    add_48: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_40: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_120, getitem_81)
    mul_191: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_80: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2]);  getitem_81 = None
    squeeze_81: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2]);  rsqrt_40 = None
    unsqueeze_40: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_121, -1)
    mul_192: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_40);  mul_191 = unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_122: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_192, [384, 64, 3, 3]);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_56: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_189, view_122, primals_123, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_20: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_56)
    sigmoid_45: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_56)
    mul_193: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_56, sigmoid_45);  convolution_56 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_123: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_124, [1, 384, -1]);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_194: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_125, 0.07450538873672485);  primals_125 = None
    view_124: "f32[384]" = torch.ops.aten.view.default(mul_194, [-1]);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [0, 2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 384, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 384, 1]" = var_mean_41[1];  var_mean_41 = None
    add_49: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_41: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_123, getitem_83)
    mul_195: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_82: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2]);  getitem_83 = None
    squeeze_83: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2]);  rsqrt_41 = None
    unsqueeze_41: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_124, -1)
    mul_196: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_41);  mul_195 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_125: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_196, [384, 64, 3, 3]);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_57: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_193, view_125, primals_126, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_46: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_57)
    mul_197: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, sigmoid_46);  sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_126: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_127, [1, 1536, -1]);  primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_198: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_128, 0.09125009274634042);  primals_128 = None
    view_127: "f32[1536]" = torch.ops.aten.view.default(mul_198, [-1]);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [0, 2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1536, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1536, 1]" = var_mean_42[1];  var_mean_42 = None
    add_50: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_42: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_126, getitem_85)
    mul_199: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_84: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2]);  getitem_85 = None
    squeeze_85: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2]);  rsqrt_42 = None
    unsqueeze_42: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_127, -1)
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
    sigmoid_47: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None
    alias_17: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_201: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_202: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_201, 2.0);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_203: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, 0.2);  mul_202 = None
    add_51: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, add_46);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_48: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_51)
    mul_204: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_51, sigmoid_48);  sigmoid_48 = None
    mul_205: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_204, 0.8980265101338745);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_2: "f32[8, 1536, 7, 7]" = torch.ops.aten.avg_pool2d.default(mul_205, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_129: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(primals_130, [1, 1536, -1]);  primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_206: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_131, 0.04562504637317021);  primals_131 = None
    view_130: "f32[1536]" = torch.ops.aten.view.default(mul_206, [-1]);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [0, 2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1536, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1536, 1]" = var_mean_43[1];  var_mean_43 = None
    add_52: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_43: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, getitem_87)
    mul_207: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_86: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2]);  getitem_87 = None
    squeeze_87: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2]);  rsqrt_43 = None
    unsqueeze_43: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_130, -1)
    mul_208: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_43);  mul_207 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_131: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_208, [1536, 1536, 1, 1]);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_61: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(avg_pool2d_2, view_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_132: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_133, [1, 384, -1]);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_209: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_134, 0.04562504637317021);  primals_134 = None
    view_133: "f32[384]" = torch.ops.aten.view.default(mul_209, [-1]);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [0, 2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 384, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 384, 1]" = var_mean_44[1];  var_mean_44 = None
    add_53: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_44: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_132, getitem_89)
    mul_210: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_88: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2]);  getitem_89 = None
    squeeze_89: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2]);  rsqrt_44 = None
    unsqueeze_44: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_133, -1)
    mul_211: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_44);  mul_210 = unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_134: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_211, [384, 1536, 1, 1]);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_62: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_205, view_134, primals_135, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_21: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_62)
    sigmoid_49: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_62)
    mul_212: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_62, sigmoid_49);  convolution_62 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_135: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_136, [1, 384, -1]);  primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_213: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_137, 0.07450538873672485);  primals_137 = None
    view_136: "f32[384]" = torch.ops.aten.view.default(mul_213, [-1]);  mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [0, 2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 384, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 384, 1]" = var_mean_45[1];  var_mean_45 = None
    add_54: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_45: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_135, getitem_91)
    mul_214: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_90: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2]);  getitem_91 = None
    squeeze_91: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2]);  rsqrt_45 = None
    unsqueeze_45: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_136, -1)
    mul_215: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_45);  mul_214 = unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_137: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_215, [384, 64, 3, 3]);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_63: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_212, view_137, primals_138, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_22: "f32[8, 384, 7, 7]" = torch.ops.aten.clone.default(convolution_63)
    sigmoid_50: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_63)
    mul_216: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_63, sigmoid_50);  convolution_63 = sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_138: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_139, [1, 384, -1]);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_217: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_140, 0.07450538873672485);  primals_140 = None
    view_139: "f32[384]" = torch.ops.aten.view.default(mul_217, [-1]);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [0, 2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 384, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 384, 1]" = var_mean_46[1];  var_mean_46 = None
    add_55: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_46: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_138, getitem_93)
    mul_218: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_92: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2]);  getitem_93 = None
    squeeze_93: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2]);  rsqrt_46 = None
    unsqueeze_46: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_139, -1)
    mul_219: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_46);  mul_218 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_140: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_219, [384, 64, 3, 3]);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_64: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_216, view_140, primals_141, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_51: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_64)
    mul_220: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, sigmoid_51);  sigmoid_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_141: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_142, [1, 1536, -1]);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_221: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_143, 0.09125009274634042);  primals_143 = None
    view_142: "f32[1536]" = torch.ops.aten.view.default(mul_221, [-1]);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [0, 2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 1536, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 1536, 1]" = var_mean_47[1];  var_mean_47 = None
    add_56: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_47: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_141, getitem_95)
    mul_222: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_94: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2]);  getitem_95 = None
    squeeze_95: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2]);  rsqrt_47 = None
    unsqueeze_47: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_142, -1)
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
    sigmoid_52: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    alias_19: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_224: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_225: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_224, 2.0);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_226: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_225, 0.2);  mul_225 = None
    add_57: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_226, convolution_61);  mul_226 = convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_53: "f32[8, 1536, 7, 7]" = torch.ops.aten.sigmoid.default(add_57)
    mul_227: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_57, sigmoid_53);  sigmoid_53 = None
    mul_228: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_227, 0.9805806756909201);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_144: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_145, [1, 384, -1]);  primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_229: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_146, 0.04562504637317021);  primals_146 = None
    view_145: "f32[384]" = torch.ops.aten.view.default(mul_229, [-1]);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [0, 2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 384, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 384, 1]" = var_mean_48[1];  var_mean_48 = None
    add_58: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_48: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_144, getitem_97)
    mul_230: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_96: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2]);  getitem_97 = None
    squeeze_97: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2]);  rsqrt_48 = None
    unsqueeze_48: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_145, -1)
    mul_231: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_48);  mul_230 = unsqueeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_146: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_231, [384, 1536, 1, 1]);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_68: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_228, view_146, primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_23: "f32[8, 384, 7, 7]" = torch.ops.aten.clone.default(convolution_68)
    sigmoid_54: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_68)
    mul_232: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, sigmoid_54);  convolution_68 = sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_147: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_148, [1, 384, -1]);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_233: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_149, 0.07450538873672485);  primals_149 = None
    view_148: "f32[384]" = torch.ops.aten.view.default(mul_233, [-1]);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [0, 2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 384, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 384, 1]" = var_mean_49[1];  var_mean_49 = None
    add_59: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_49: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_147, getitem_99)
    mul_234: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_98: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2]);  getitem_99 = None
    squeeze_99: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2]);  rsqrt_49 = None
    unsqueeze_49: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_148, -1)
    mul_235: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_49);  mul_234 = unsqueeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_149: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_235, [384, 64, 3, 3]);  mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_69: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_232, view_149, primals_150, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_24: "f32[8, 384, 7, 7]" = torch.ops.aten.clone.default(convolution_69)
    sigmoid_55: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_69)
    mul_236: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_69, sigmoid_55);  convolution_69 = sigmoid_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_150: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_151, [1, 384, -1]);  primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_237: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_152, 0.07450538873672485);  primals_152 = None
    view_151: "f32[384]" = torch.ops.aten.view.default(mul_237, [-1]);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [0, 2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 384, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 384, 1]" = var_mean_50[1];  var_mean_50 = None
    add_60: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_50: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_150, getitem_101)
    mul_238: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_100: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2]);  getitem_101 = None
    squeeze_101: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2]);  rsqrt_50 = None
    unsqueeze_50: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_151, -1)
    mul_239: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_50);  mul_238 = unsqueeze_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_152: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_239, [384, 64, 3, 3]);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_70: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_236, view_152, primals_153, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_56: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_70)
    mul_240: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, sigmoid_56);  sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_153: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_154, [1, 1536, -1]);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_155, 0.09125009274634042);  primals_155 = None
    view_154: "f32[1536]" = torch.ops.aten.view.default(mul_241, [-1]);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [0, 2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1536, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1536, 1]" = var_mean_51[1];  var_mean_51 = None
    add_61: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_51: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_153, getitem_103)
    mul_242: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_102: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2]);  getitem_103 = None
    squeeze_103: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2]);  rsqrt_51 = None
    unsqueeze_51: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_154, -1)
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
    sigmoid_57: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    alias_21: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_244: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_245: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_244, 2.0);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_246: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_245, 0.2);  mul_245 = None
    add_62: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_246, add_57);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_58: "f32[8, 1536, 7, 7]" = torch.ops.aten.sigmoid.default(add_62)
    mul_247: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_62, sigmoid_58);  sigmoid_58 = None
    mul_248: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_247, 0.9622504486493761);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_156: "f32[1, 384, 1536]" = torch.ops.aten.view.default(primals_157, [1, 384, -1]);  primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_249: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_158, 0.04562504637317021);  primals_158 = None
    view_157: "f32[384]" = torch.ops.aten.view.default(mul_249, [-1]);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [0, 2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 384, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 384, 1]" = var_mean_52[1];  var_mean_52 = None
    add_63: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_52: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_156, getitem_105)
    mul_250: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_104: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2]);  getitem_105 = None
    squeeze_105: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2]);  rsqrt_52 = None
    unsqueeze_52: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_157, -1)
    mul_251: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_52);  mul_250 = unsqueeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_158: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_251, [384, 1536, 1, 1]);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_74: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_248, view_158, primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    clone_25: "f32[8, 384, 7, 7]" = torch.ops.aten.clone.default(convolution_74)
    sigmoid_59: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_74)
    mul_252: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, sigmoid_59);  convolution_74 = sigmoid_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_159: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_160, [1, 384, -1]);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_253: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_161, 0.07450538873672485);  primals_161 = None
    view_160: "f32[384]" = torch.ops.aten.view.default(mul_253, [-1]);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [0, 2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 384, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 384, 1]" = var_mean_53[1];  var_mean_53 = None
    add_64: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_53: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_159, getitem_107)
    mul_254: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_106: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2]);  getitem_107 = None
    squeeze_107: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2]);  rsqrt_53 = None
    unsqueeze_53: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_160, -1)
    mul_255: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_53);  mul_254 = unsqueeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_161: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_255, [384, 64, 3, 3]);  mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_75: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_252, view_161, primals_162, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    clone_26: "f32[8, 384, 7, 7]" = torch.ops.aten.clone.default(convolution_75)
    sigmoid_60: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_75)
    mul_256: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_75, sigmoid_60);  convolution_75 = sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_162: "f32[1, 384, 576]" = torch.ops.aten.view.default(primals_163, [1, 384, -1]);  primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_257: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_164, 0.07450538873672485);  primals_164 = None
    view_163: "f32[384]" = torch.ops.aten.view.default(mul_257, [-1]);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [0, 2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 384, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 384, 1]" = var_mean_54[1];  var_mean_54 = None
    add_65: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_54: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, getitem_109)
    mul_258: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_108: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2]);  getitem_109 = None
    squeeze_109: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2]);  rsqrt_54 = None
    unsqueeze_54: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_163, -1)
    mul_259: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_54);  mul_258 = unsqueeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_164: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_259, [384, 64, 3, 3]);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_76: "f32[8, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_256, view_164, primals_165, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_61: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_76)
    mul_260: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_76, sigmoid_61);  sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_165: "f32[1, 1536, 384]" = torch.ops.aten.view.default(primals_166, [1, 1536, -1]);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_261: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_167, 0.09125009274634042);  primals_167 = None
    view_166: "f32[1536]" = torch.ops.aten.view.default(mul_261, [-1]);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [0, 2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1536, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 1536, 1]" = var_mean_55[1];  var_mean_55 = None
    add_66: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_55: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_165, getitem_111)
    mul_262: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_110: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2]);  getitem_111 = None
    squeeze_111: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2]);  rsqrt_55 = None
    unsqueeze_55: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_166, -1)
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
    sigmoid_62: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None
    alias_23: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_264: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_265: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_264, 2.0);  mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_266: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_265, 0.2);  mul_265 = None
    add_67: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_266, add_62);  mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_168: "f32[1, 2304, 1536]" = torch.ops.aten.view.default(primals_169, [1, 2304, -1]);  primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_267: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_170, 0.04562504637317021);  primals_170 = None
    view_169: "f32[2304]" = torch.ops.aten.view.default(mul_267, [-1]);  mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [0, 2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 2304, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 2304, 1]" = var_mean_56[1];  var_mean_56 = None
    add_68: "f32[1, 2304, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 2304, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_56: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, getitem_113)
    mul_268: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_112: "f32[2304]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2]);  getitem_113 = None
    squeeze_113: "f32[2304]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2]);  rsqrt_56 = None
    unsqueeze_56: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(view_169, -1)
    mul_269: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_56);  mul_268 = unsqueeze_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_170: "f32[2304, 1536, 1, 1]" = torch.ops.aten.view.default(mul_269, [2304, 1536, 1, 1]);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_80: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_67, view_170, primals_171, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:445, code: x = self.final_act(x)
    clone_27: "f32[8, 2304, 7, 7]" = torch.ops.aten.clone.default(convolution_80)
    sigmoid_63: "f32[8, 2304, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_80)
    mul_270: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_80, sigmoid_63);  convolution_80 = sigmoid_63 = None
    
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
    mm: "f32[8, 2304]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2304]" = torch.ops.aten.mm.default(permute_2, clone_28);  permute_2 = clone_28 = None
    permute_3: "f32[2304, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_172: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2304]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_173: "f32[8, 2304, 1, 1]" = torch.ops.aten.view.default(mm, [8, 2304, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2304, 7, 7]" = torch.ops.aten.expand.default(view_173, [8, 2304, 7, 7]);  view_173 = None
    div: "f32[8, 2304, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:445, code: x = self.final_act(x)
    sigmoid_64: "f32[8, 2304, 7, 7]" = torch.ops.aten.sigmoid.default(clone_27)
    full: "f32[8, 2304, 7, 7]" = torch.ops.aten.full.default([8, 2304, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_57: "f32[8, 2304, 7, 7]" = torch.ops.aten.sub.Tensor(full, sigmoid_64);  full = None
    mul_271: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(clone_27, sub_57);  clone_27 = sub_57 = None
    add_69: "f32[8, 2304, 7, 7]" = torch.ops.aten.add.Scalar(mul_271, 1);  mul_271 = None
    mul_272: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_69);  sigmoid_64 = add_69 = None
    mul_273: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(div, mul_272);  div = mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_2: "f32[2304]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_273, add_67, view_170, [2304], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_273 = add_67 = view_170 = None
    getitem_114: "f32[8, 1536, 7, 7]" = convolution_backward[0]
    getitem_115: "f32[2304, 1536, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_174: "f32[1, 2304, 1536]" = torch.ops.aten.view.default(getitem_115, [1, 2304, 1536]);  getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_57: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(squeeze_112, 0);  squeeze_112 = None
    unsqueeze_58: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
    sum_3: "f32[2304]" = torch.ops.aten.sum.dim_IntList(view_174, [0, 2])
    sub_58: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58)
    mul_274: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(view_174, sub_58);  sub_58 = None
    sum_4: "f32[2304]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 2]);  mul_274 = None
    mul_275: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_3, 0.0006510416666666666);  sum_3 = None
    unsqueeze_59: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_275, 0);  mul_275 = None
    unsqueeze_60: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 2);  unsqueeze_59 = None
    mul_276: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_4, 0.0006510416666666666)
    mul_277: "f32[2304]" = torch.ops.aten.mul.Tensor(squeeze_113, squeeze_113)
    mul_278: "f32[2304]" = torch.ops.aten.mul.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    unsqueeze_61: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
    unsqueeze_62: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
    mul_279: "f32[2304]" = torch.ops.aten.mul.Tensor(squeeze_113, view_169);  view_169 = None
    unsqueeze_63: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_279, 0);  mul_279 = None
    unsqueeze_64: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
    sub_59: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58);  view_168 = unsqueeze_58 = None
    mul_280: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_62);  sub_59 = unsqueeze_62 = None
    sub_60: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_174, mul_280);  view_174 = mul_280 = None
    sub_61: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_60);  sub_60 = unsqueeze_60 = None
    mul_281: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_64);  sub_61 = unsqueeze_64 = None
    mul_282: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_113);  sum_4 = squeeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_175: "f32[2304, 1, 1, 1]" = torch.ops.aten.view.default(mul_282, [2304, 1, 1, 1]);  mul_282 = None
    mul_283: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_175, 0.04562504637317021);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_176: "f32[2304, 1536, 1, 1]" = torch.ops.aten.view.default(mul_281, [2304, 1536, 1, 1]);  mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_284: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_114, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_285: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, 2.0);  mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_286: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_285, convolution_77);  convolution_77 = None
    mul_287: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_285, sigmoid_62);  mul_285 = sigmoid_62 = None
    sum_5: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2, 3], True);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_24: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    sub_62: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_24)
    mul_288: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_24, sub_62);  alias_24 = sub_62 = None
    mul_289: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_5, mul_288);  sum_5 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_6: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_289, relu_11, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = primals_218 = None
    getitem_117: "f32[8, 384, 1, 1]" = convolution_backward_1[0]
    getitem_118: "f32[1536, 384, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_26: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_27: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_27, 0);  alias_27 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le, scalar_tensor, getitem_117);  le = scalar_tensor = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_7: "f32[384]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, mean_11, primals_216, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_11 = primals_216 = None
    getitem_120: "f32[8, 1536, 1, 1]" = convolution_backward_2[0]
    getitem_121: "f32[384, 1536, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1536, 7, 7]" = torch.ops.aten.expand.default(getitem_120, [8, 1536, 7, 7]);  getitem_120 = None
    div_1: "f32[8, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_70: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_287, div_1);  mul_287 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_8: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_70, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_70, mul_260, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_70 = mul_260 = view_167 = None
    getitem_123: "f32[8, 384, 7, 7]" = convolution_backward_3[0]
    getitem_124: "f32[1536, 384, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_177: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_124, [1, 1536, 384]);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_65: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_110, 0);  squeeze_110 = None
    unsqueeze_66: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
    sum_9: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_177, [0, 2])
    sub_63: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66)
    mul_290: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_177, sub_63);  sub_63 = None
    sum_10: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 2]);  mul_290 = None
    mul_291: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_9, 0.0026041666666666665);  sum_9 = None
    unsqueeze_67: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_68: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 2);  unsqueeze_67 = None
    mul_292: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_10, 0.0026041666666666665)
    mul_293: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, squeeze_111)
    mul_294: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    unsqueeze_69: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_70: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
    mul_295: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, view_166);  view_166 = None
    unsqueeze_71: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
    unsqueeze_72: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
    sub_64: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66);  view_165 = unsqueeze_66 = None
    mul_296: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_70);  sub_64 = unsqueeze_70 = None
    sub_65: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_177, mul_296);  view_177 = mul_296 = None
    sub_66: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_68);  sub_65 = unsqueeze_68 = None
    mul_297: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_72);  sub_66 = unsqueeze_72 = None
    mul_298: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_111);  sum_10 = squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_298, [1536, 1, 1, 1]);  mul_298 = None
    mul_299: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_178, 0.09125009274634042);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_179: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_297, [1536, 384, 1, 1]);  mul_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_65: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_76)
    full_1: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_67: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_1, sigmoid_65);  full_1 = None
    mul_300: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_76, sub_67);  convolution_76 = sub_67 = None
    add_71: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_300, 1);  mul_300 = None
    mul_301: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_71);  sigmoid_65 = add_71 = None
    mul_302: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_123, mul_301);  getitem_123 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_11: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_302, mul_256, view_164, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_302 = mul_256 = view_164 = None
    getitem_126: "f32[8, 384, 7, 7]" = convolution_backward_4[0]
    getitem_127: "f32[384, 64, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_180: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_127, [1, 384, 576]);  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_73: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_74: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 2);  unsqueeze_73 = None
    sum_12: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_180, [0, 2])
    sub_68: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74)
    mul_303: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_180, sub_68);  sub_68 = None
    sum_13: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2]);  mul_303 = None
    mul_304: "f32[384]" = torch.ops.aten.mul.Tensor(sum_12, 0.001736111111111111);  sum_12 = None
    unsqueeze_75: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
    unsqueeze_76: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
    mul_305: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, 0.001736111111111111)
    mul_306: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_307: "f32[384]" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_77: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
    unsqueeze_78: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
    mul_308: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_109, view_163);  view_163 = None
    unsqueeze_79: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_80: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 2);  unsqueeze_79 = None
    sub_69: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74);  view_162 = unsqueeze_74 = None
    mul_309: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_78);  sub_69 = unsqueeze_78 = None
    sub_70: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_180, mul_309);  view_180 = mul_309 = None
    sub_71: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_76);  sub_70 = unsqueeze_76 = None
    mul_310: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_80);  sub_71 = unsqueeze_80 = None
    mul_311: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_109);  sum_13 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_181: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_311, [384, 1, 1, 1]);  mul_311 = None
    mul_312: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_181, 0.07450538873672485);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_182: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_310, [384, 64, 3, 3]);  mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_66: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(clone_26)
    full_2: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_72: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_2, sigmoid_66);  full_2 = None
    mul_313: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(clone_26, sub_72);  clone_26 = sub_72 = None
    add_72: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_313, 1);  mul_313 = None
    mul_314: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_72);  sigmoid_66 = add_72 = None
    mul_315: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_126, mul_314);  getitem_126 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_14: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_315, mul_252, view_161, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_315 = mul_252 = view_161 = None
    getitem_129: "f32[8, 384, 7, 7]" = convolution_backward_5[0]
    getitem_130: "f32[384, 64, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_183: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_130, [1, 384, 576]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_81: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_106, 0);  squeeze_106 = None
    unsqueeze_82: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
    sum_15: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_183, [0, 2])
    sub_73: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82)
    mul_316: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_183, sub_73);  sub_73 = None
    sum_16: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 2]);  mul_316 = None
    mul_317: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, 0.001736111111111111);  sum_15 = None
    unsqueeze_83: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
    unsqueeze_84: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
    mul_318: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, 0.001736111111111111)
    mul_319: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_107, squeeze_107)
    mul_320: "f32[384]" = torch.ops.aten.mul.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    unsqueeze_85: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_86: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 2);  unsqueeze_85 = None
    mul_321: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_107, view_160);  view_160 = None
    unsqueeze_87: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
    unsqueeze_88: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
    sub_74: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82);  view_159 = unsqueeze_82 = None
    mul_322: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_86);  sub_74 = unsqueeze_86 = None
    sub_75: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_183, mul_322);  view_183 = mul_322 = None
    sub_76: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_84);  sub_75 = unsqueeze_84 = None
    mul_323: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_88);  sub_76 = unsqueeze_88 = None
    mul_324: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_107);  sum_16 = squeeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_184: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_324, [384, 1, 1, 1]);  mul_324 = None
    mul_325: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_184, 0.07450538873672485);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_185: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_323, [384, 64, 3, 3]);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_67: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(clone_25)
    full_3: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_77: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_3, sigmoid_67);  full_3 = None
    mul_326: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(clone_25, sub_77);  clone_25 = sub_77 = None
    add_73: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_326, 1);  mul_326 = None
    mul_327: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_73);  sigmoid_67 = add_73 = None
    mul_328: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_129, mul_327);  getitem_129 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_328, mul_248, view_158, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = mul_248 = view_158 = None
    getitem_132: "f32[8, 1536, 7, 7]" = convolution_backward_6[0]
    getitem_133: "f32[384, 1536, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_186: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_133, [1, 384, 1536]);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_89: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_104, 0);  squeeze_104 = None
    unsqueeze_90: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
    sum_18: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_186, [0, 2])
    sub_78: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90)
    mul_329: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_186, sub_78);  sub_78 = None
    sum_19: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2]);  mul_329 = None
    mul_330: "f32[384]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006510416666666666);  sum_18 = None
    unsqueeze_91: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_92: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None
    mul_331: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006510416666666666)
    mul_332: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_105, squeeze_105)
    mul_333: "f32[384]" = torch.ops.aten.mul.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None
    unsqueeze_93: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
    unsqueeze_94: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
    mul_334: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_105, view_157);  view_157 = None
    unsqueeze_95: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
    unsqueeze_96: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
    sub_79: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90);  view_156 = unsqueeze_90 = None
    mul_335: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_94);  sub_79 = unsqueeze_94 = None
    sub_80: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_186, mul_335);  view_186 = mul_335 = None
    sub_81: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_92);  sub_80 = unsqueeze_92 = None
    mul_336: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_96);  sub_81 = unsqueeze_96 = None
    mul_337: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_105);  sum_19 = squeeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_187: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_337, [384, 1, 1, 1]);  mul_337 = None
    mul_338: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_187, 0.04562504637317021);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_188: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_336, [384, 1536, 1, 1]);  mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_339: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_132, 0.9622504486493761);  getitem_132 = None
    sigmoid_68: "f32[8, 1536, 7, 7]" = torch.ops.aten.sigmoid.default(add_62)
    full_4: "f32[8, 1536, 7, 7]" = torch.ops.aten.full.default([8, 1536, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_82: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(full_4, sigmoid_68);  full_4 = None
    mul_340: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_62, sub_82);  add_62 = sub_82 = None
    add_74: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Scalar(mul_340, 1);  mul_340 = None
    mul_341: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_74);  sigmoid_68 = add_74 = None
    mul_342: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_339, mul_341);  mul_339 = mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_75: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(getitem_114, mul_342);  getitem_114 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_343: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_75, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_344: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_343, 2.0);  mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_345: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_344, convolution_71);  convolution_71 = None
    mul_346: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_344, sigmoid_57);  mul_344 = sigmoid_57 = None
    sum_20: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2, 3], True);  mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_28: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    sub_83: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_347: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_83);  alias_28 = sub_83 = None
    mul_348: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, mul_347);  sum_20 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_21: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_348, relu_10, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_348 = primals_214 = None
    getitem_135: "f32[8, 384, 1, 1]" = convolution_backward_7[0]
    getitem_136: "f32[1536, 384, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_30: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_31: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_1: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_135);  le_1 = scalar_tensor_1 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_22: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_1, mean_10, primals_212, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_10 = primals_212 = None
    getitem_138: "f32[8, 1536, 1, 1]" = convolution_backward_8[0]
    getitem_139: "f32[384, 1536, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 1536, 7, 7]" = torch.ops.aten.expand.default(getitem_138, [8, 1536, 7, 7]);  getitem_138 = None
    div_2: "f32[8, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_76: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_346, div_2);  mul_346 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_23: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_76, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(add_76, mul_240, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_76 = mul_240 = view_155 = None
    getitem_141: "f32[8, 384, 7, 7]" = convolution_backward_9[0]
    getitem_142: "f32[1536, 384, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_189: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_142, [1, 1536, 384]);  getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_97: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_98: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 2);  unsqueeze_97 = None
    sum_24: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_189, [0, 2])
    sub_84: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98)
    mul_349: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_189, sub_84);  sub_84 = None
    sum_25: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 2]);  mul_349 = None
    mul_350: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_24, 0.0026041666666666665);  sum_24 = None
    unsqueeze_99: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
    unsqueeze_100: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
    mul_351: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_25, 0.0026041666666666665)
    mul_352: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_353: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    unsqueeze_101: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_102: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
    mul_354: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, view_154);  view_154 = None
    unsqueeze_103: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_354, 0);  mul_354 = None
    unsqueeze_104: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 2);  unsqueeze_103 = None
    sub_85: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98);  view_153 = unsqueeze_98 = None
    mul_355: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_102);  sub_85 = unsqueeze_102 = None
    sub_86: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_189, mul_355);  view_189 = mul_355 = None
    sub_87: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_100);  sub_86 = unsqueeze_100 = None
    mul_356: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_104);  sub_87 = unsqueeze_104 = None
    mul_357: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_103);  sum_25 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_190: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_357, [1536, 1, 1, 1]);  mul_357 = None
    mul_358: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_190, 0.09125009274634042);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_191: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_356, [1536, 384, 1, 1]);  mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_69: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_70)
    full_5: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_88: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_5, sigmoid_69);  full_5 = None
    mul_359: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, sub_88);  convolution_70 = sub_88 = None
    add_77: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_359, 1);  mul_359 = None
    mul_360: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_77);  sigmoid_69 = add_77 = None
    mul_361: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_141, mul_360);  getitem_141 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_361, mul_236, view_152, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_361 = mul_236 = view_152 = None
    getitem_144: "f32[8, 384, 7, 7]" = convolution_backward_10[0]
    getitem_145: "f32[384, 64, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_192: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_145, [1, 384, 576]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_105: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_100, 0);  squeeze_100 = None
    unsqueeze_106: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
    sum_27: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_192, [0, 2])
    sub_89: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106)
    mul_362: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_192, sub_89);  sub_89 = None
    sum_28: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 2]);  mul_362 = None
    mul_363: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, 0.001736111111111111);  sum_27 = None
    unsqueeze_107: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_363, 0);  mul_363 = None
    unsqueeze_108: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
    mul_364: "f32[384]" = torch.ops.aten.mul.Tensor(sum_28, 0.001736111111111111)
    mul_365: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_101, squeeze_101)
    mul_366: "f32[384]" = torch.ops.aten.mul.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    unsqueeze_109: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_366, 0);  mul_366 = None
    unsqueeze_110: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 2);  unsqueeze_109 = None
    mul_367: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_101, view_151);  view_151 = None
    unsqueeze_111: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
    unsqueeze_112: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    sub_90: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106);  view_150 = unsqueeze_106 = None
    mul_368: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_110);  sub_90 = unsqueeze_110 = None
    sub_91: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_192, mul_368);  view_192 = mul_368 = None
    sub_92: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_108);  sub_91 = unsqueeze_108 = None
    mul_369: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_112);  sub_92 = unsqueeze_112 = None
    mul_370: "f32[384]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_101);  sum_28 = squeeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_193: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_370, [384, 1, 1, 1]);  mul_370 = None
    mul_371: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_193, 0.07450538873672485);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_194: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_369, [384, 64, 3, 3]);  mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_70: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(clone_24)
    full_6: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_93: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_6, sigmoid_70);  full_6 = None
    mul_372: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(clone_24, sub_93);  clone_24 = sub_93 = None
    add_78: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_372, 1);  mul_372 = None
    mul_373: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_78);  sigmoid_70 = add_78 = None
    mul_374: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_144, mul_373);  getitem_144 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_29: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_374, mul_232, view_149, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_374 = mul_232 = view_149 = None
    getitem_147: "f32[8, 384, 7, 7]" = convolution_backward_11[0]
    getitem_148: "f32[384, 64, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_195: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_148, [1, 384, 576]);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_113: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_98, 0);  squeeze_98 = None
    unsqueeze_114: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
    sum_30: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_195, [0, 2])
    sub_94: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114)
    mul_375: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_195, sub_94);  sub_94 = None
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2]);  mul_375 = None
    mul_376: "f32[384]" = torch.ops.aten.mul.Tensor(sum_30, 0.001736111111111111);  sum_30 = None
    unsqueeze_115: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_116: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
    mul_377: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, 0.001736111111111111)
    mul_378: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_99, squeeze_99)
    mul_379: "f32[384]" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    unsqueeze_117: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_118: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    mul_380: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_99, view_148);  view_148 = None
    unsqueeze_119: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
    unsqueeze_120: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    sub_95: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114);  view_147 = unsqueeze_114 = None
    mul_381: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_118);  sub_95 = unsqueeze_118 = None
    sub_96: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_195, mul_381);  view_195 = mul_381 = None
    sub_97: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_116);  sub_96 = unsqueeze_116 = None
    mul_382: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_120);  sub_97 = unsqueeze_120 = None
    mul_383: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_99);  sum_31 = squeeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_196: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_383, [384, 1, 1, 1]);  mul_383 = None
    mul_384: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_196, 0.07450538873672485);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_197: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_382, [384, 64, 3, 3]);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_71: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(clone_23)
    full_7: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_98: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_7, sigmoid_71);  full_7 = None
    mul_385: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(clone_23, sub_98);  clone_23 = sub_98 = None
    add_79: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_385, 1);  mul_385 = None
    mul_386: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_79);  sigmoid_71 = add_79 = None
    mul_387: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_147, mul_386);  getitem_147 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_387, mul_228, view_146, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_387 = mul_228 = view_146 = None
    getitem_150: "f32[8, 1536, 7, 7]" = convolution_backward_12[0]
    getitem_151: "f32[384, 1536, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_198: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_151, [1, 384, 1536]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_121: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_122: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
    sum_33: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_198, [0, 2])
    sub_99: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122)
    mul_388: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_198, sub_99);  sub_99 = None
    sum_34: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2]);  mul_388 = None
    mul_389: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006510416666666666);  sum_33 = None
    unsqueeze_123: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
    unsqueeze_124: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    mul_390: "f32[384]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006510416666666666)
    mul_391: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_392: "f32[384]" = torch.ops.aten.mul.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    unsqueeze_125: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_126: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    mul_393: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, view_145);  view_145 = None
    unsqueeze_127: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
    unsqueeze_128: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    sub_100: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122);  view_144 = unsqueeze_122 = None
    mul_394: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_126);  sub_100 = unsqueeze_126 = None
    sub_101: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_198, mul_394);  view_198 = mul_394 = None
    sub_102: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_124);  sub_101 = unsqueeze_124 = None
    mul_395: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_128);  sub_102 = unsqueeze_128 = None
    mul_396: "f32[384]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_97);  sum_34 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_199: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_396, [384, 1, 1, 1]);  mul_396 = None
    mul_397: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_199, 0.04562504637317021);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_200: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_395, [384, 1536, 1, 1]);  mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_398: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_150, 0.9805806756909201);  getitem_150 = None
    sigmoid_72: "f32[8, 1536, 7, 7]" = torch.ops.aten.sigmoid.default(add_57)
    full_8: "f32[8, 1536, 7, 7]" = torch.ops.aten.full.default([8, 1536, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_103: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(full_8, sigmoid_72);  full_8 = None
    mul_399: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_57, sub_103);  add_57 = sub_103 = None
    add_80: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Scalar(mul_399, 1);  mul_399 = None
    mul_400: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_80);  sigmoid_72 = add_80 = None
    mul_401: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_398, mul_400);  mul_398 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_81: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(add_75, mul_401);  add_75 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_402: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_81, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_403: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_402, 2.0);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_404: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, convolution_65);  convolution_65 = None
    mul_405: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, sigmoid_52);  mul_403 = sigmoid_52 = None
    sum_35: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2, 3], True);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_32: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    sub_104: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_32)
    mul_406: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_32, sub_104);  alias_32 = sub_104 = None
    mul_407: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_35, mul_406);  sum_35 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_36: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_407, relu_9, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = primals_210 = None
    getitem_153: "f32[8, 384, 1, 1]" = convolution_backward_13[0]
    getitem_154: "f32[1536, 384, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_34: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_35: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_2: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_35, 0);  alias_35 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_153);  le_2 = scalar_tensor_2 = getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_37: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_2, mean_9, primals_208, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = mean_9 = primals_208 = None
    getitem_156: "f32[8, 1536, 1, 1]" = convolution_backward_14[0]
    getitem_157: "f32[384, 1536, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 1536, 7, 7]" = torch.ops.aten.expand.default(getitem_156, [8, 1536, 7, 7]);  getitem_156 = None
    div_3: "f32[8, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_82: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_405, div_3);  mul_405 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_38: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_82, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(add_82, mul_220, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_82 = mul_220 = view_143 = None
    getitem_159: "f32[8, 384, 7, 7]" = convolution_backward_15[0]
    getitem_160: "f32[1536, 384, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_201: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_160, [1, 1536, 384]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_129: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_94, 0);  squeeze_94 = None
    unsqueeze_130: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    sum_39: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 2])
    sub_105: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130)
    mul_408: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_201, sub_105);  sub_105 = None
    sum_40: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2]);  mul_408 = None
    mul_409: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_39, 0.0026041666666666665);  sum_39 = None
    unsqueeze_131: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_132: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    mul_410: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_40, 0.0026041666666666665)
    mul_411: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, squeeze_95)
    mul_412: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_133: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_134: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    mul_413: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, view_142);  view_142 = None
    unsqueeze_135: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_136: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    sub_106: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130);  view_141 = unsqueeze_130 = None
    mul_414: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_134);  sub_106 = unsqueeze_134 = None
    sub_107: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_201, mul_414);  view_201 = mul_414 = None
    sub_108: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_132);  sub_107 = unsqueeze_132 = None
    mul_415: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_136);  sub_108 = unsqueeze_136 = None
    mul_416: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_95);  sum_40 = squeeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_202: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_416, [1536, 1, 1, 1]);  mul_416 = None
    mul_417: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_202, 0.09125009274634042);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_203: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_415, [1536, 384, 1, 1]);  mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_73: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_64)
    full_9: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_109: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_9, sigmoid_73);  full_9 = None
    mul_418: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, sub_109);  convolution_64 = sub_109 = None
    add_83: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_418, 1);  mul_418 = None
    mul_419: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_83);  sigmoid_73 = add_83 = None
    mul_420: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_159, mul_419);  getitem_159 = mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_420, mul_216, view_140, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_420 = mul_216 = view_140 = None
    getitem_162: "f32[8, 384, 7, 7]" = convolution_backward_16[0]
    getitem_163: "f32[384, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_204: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_163, [1, 384, 576]);  getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_137: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_92, 0);  squeeze_92 = None
    unsqueeze_138: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    sum_42: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_204, [0, 2])
    sub_110: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138)
    mul_421: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_204, sub_110);  sub_110 = None
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 2]);  mul_421 = None
    mul_422: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, 0.001736111111111111);  sum_42 = None
    unsqueeze_139: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_140: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    mul_423: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, 0.001736111111111111)
    mul_424: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_93, squeeze_93)
    mul_425: "f32[384]" = torch.ops.aten.mul.Tensor(mul_423, mul_424);  mul_423 = mul_424 = None
    unsqueeze_141: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
    unsqueeze_142: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    mul_426: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_93, view_139);  view_139 = None
    unsqueeze_143: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
    unsqueeze_144: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    sub_111: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138);  view_138 = unsqueeze_138 = None
    mul_427: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_142);  sub_111 = unsqueeze_142 = None
    sub_112: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_204, mul_427);  view_204 = mul_427 = None
    sub_113: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_140);  sub_112 = unsqueeze_140 = None
    mul_428: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_144);  sub_113 = unsqueeze_144 = None
    mul_429: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_93);  sum_43 = squeeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_205: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_429, [384, 1, 1, 1]);  mul_429 = None
    mul_430: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_205, 0.07450538873672485);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_206: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_428, [384, 64, 3, 3]);  mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_74: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(clone_22)
    full_10: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_114: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_10, sigmoid_74);  full_10 = None
    mul_431: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(clone_22, sub_114);  clone_22 = sub_114 = None
    add_84: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_431, 1);  mul_431 = None
    mul_432: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_84);  sigmoid_74 = add_84 = None
    mul_433: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_162, mul_432);  getitem_162 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_433, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_433, mul_212, view_137, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_433 = mul_212 = view_137 = None
    getitem_165: "f32[8, 384, 14, 14]" = convolution_backward_17[0]
    getitem_166: "f32[384, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_207: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_166, [1, 384, 576]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_145: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_146: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_207, [0, 2])
    sub_115: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146)
    mul_434: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_207, sub_115);  sub_115 = None
    sum_46: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 2]);  mul_434 = None
    mul_435: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, 0.001736111111111111);  sum_45 = None
    unsqueeze_147: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_148: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    mul_436: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, 0.001736111111111111)
    mul_437: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_438: "f32[384]" = torch.ops.aten.mul.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
    unsqueeze_149: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
    unsqueeze_150: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    mul_439: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_91, view_136);  view_136 = None
    unsqueeze_151: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
    unsqueeze_152: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    sub_116: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146);  view_135 = unsqueeze_146 = None
    mul_440: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_150);  sub_116 = unsqueeze_150 = None
    sub_117: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_207, mul_440);  view_207 = mul_440 = None
    sub_118: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_148);  sub_117 = unsqueeze_148 = None
    mul_441: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_152);  sub_118 = unsqueeze_152 = None
    mul_442: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_91);  sum_46 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_208: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_442, [384, 1, 1, 1]);  mul_442 = None
    mul_443: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_208, 0.07450538873672485);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_209: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_441, [384, 64, 3, 3]);  mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_75: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_21)
    full_11: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_119: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_11, sigmoid_75);  full_11 = None
    mul_444: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_21, sub_119);  clone_21 = sub_119 = None
    add_85: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_444, 1);  mul_444 = None
    mul_445: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_85);  sigmoid_75 = add_85 = None
    mul_446: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_165, mul_445);  getitem_165 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_47: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_446, mul_205, view_134, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_446 = view_134 = None
    getitem_168: "f32[8, 1536, 14, 14]" = convolution_backward_18[0]
    getitem_169: "f32[384, 1536, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_210: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_169, [1, 384, 1536]);  getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_153: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_88, 0);  squeeze_88 = None
    unsqueeze_154: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    sum_48: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_210, [0, 2])
    sub_120: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154)
    mul_447: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_210, sub_120);  sub_120 = None
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2]);  mul_447 = None
    mul_448: "f32[384]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006510416666666666);  sum_48 = None
    unsqueeze_155: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_156: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    mul_449: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006510416666666666)
    mul_450: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_89, squeeze_89)
    mul_451: "f32[384]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_157: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_158: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    mul_452: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_89, view_133);  view_133 = None
    unsqueeze_159: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_160: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    sub_121: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154);  view_132 = unsqueeze_154 = None
    mul_453: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_158);  sub_121 = unsqueeze_158 = None
    sub_122: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_210, mul_453);  view_210 = mul_453 = None
    sub_123: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_156);  sub_122 = unsqueeze_156 = None
    mul_454: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_160);  sub_123 = unsqueeze_160 = None
    mul_455: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_89);  sum_49 = squeeze_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_211: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_455, [384, 1, 1, 1]);  mul_455 = None
    mul_456: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_211, 0.04562504637317021);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_212: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_454, [384, 1536, 1, 1]);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_50: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_81, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_81, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_81 = avg_pool2d_2 = view_131 = None
    getitem_171: "f32[8, 1536, 7, 7]" = convolution_backward_19[0]
    getitem_172: "f32[1536, 1536, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_213: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(getitem_172, [1, 1536, 1536]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_161: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_86, 0);  squeeze_86 = None
    unsqueeze_162: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    sum_51: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_213, [0, 2])
    sub_124: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162)
    mul_457: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(view_213, sub_124);  sub_124 = None
    sum_52: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2]);  mul_457 = None
    mul_458: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006510416666666666);  sum_51 = None
    unsqueeze_163: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_164: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    mul_459: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006510416666666666)
    mul_460: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, squeeze_87)
    mul_461: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_165: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
    unsqueeze_166: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    mul_462: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, view_130);  view_130 = None
    unsqueeze_167: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_168: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    sub_125: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162);  view_129 = unsqueeze_162 = None
    mul_463: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_166);  sub_125 = unsqueeze_166 = None
    sub_126: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_213, mul_463);  view_213 = mul_463 = None
    sub_127: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_164);  sub_126 = unsqueeze_164 = None
    mul_464: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_168);  sub_127 = unsqueeze_168 = None
    mul_465: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_87);  sum_52 = squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_214: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_465, [1536, 1, 1, 1]);  mul_465 = None
    mul_466: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_214, 0.04562504637317021);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_215: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_464, [1536, 1536, 1, 1]);  mul_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward: "f32[8, 1536, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(getitem_171, mul_205, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_171 = mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_86: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(getitem_168, avg_pool2d_backward);  getitem_168 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_467: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_86, 0.8980265101338745);  add_86 = None
    sigmoid_76: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_51)
    full_12: "f32[8, 1536, 14, 14]" = torch.ops.aten.full.default([8, 1536, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_128: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_12, sigmoid_76);  full_12 = None
    mul_468: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_51, sub_128);  add_51 = sub_128 = None
    add_87: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_468, 1);  mul_468 = None
    mul_469: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_76, add_87);  sigmoid_76 = add_87 = None
    mul_470: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_467, mul_469);  mul_467 = mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_471: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_470, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_472: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_471, 2.0);  mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_473: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_472, convolution_58);  convolution_58 = None
    mul_474: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_472, sigmoid_47);  mul_472 = sigmoid_47 = None
    sum_53: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2, 3], True);  mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_36: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    sub_129: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_36)
    mul_475: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_36, sub_129);  alias_36 = sub_129 = None
    mul_476: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_53, mul_475);  sum_53 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_54: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_476, relu_8, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_476 = primals_206 = None
    getitem_174: "f32[8, 384, 1, 1]" = convolution_backward_20[0]
    getitem_175: "f32[1536, 384, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_38: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_39: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_3: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_174);  le_3 = scalar_tensor_3 = getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_3, mean_8, primals_204, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_8 = primals_204 = None
    getitem_177: "f32[8, 1536, 1, 1]" = convolution_backward_21[0]
    getitem_178: "f32[384, 1536, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_177, [8, 1536, 14, 14]);  getitem_177 = None
    div_4: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_88: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_474, div_4);  mul_474 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_56: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_88, mul_197, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_88 = mul_197 = view_128 = None
    getitem_180: "f32[8, 384, 14, 14]" = convolution_backward_22[0]
    getitem_181: "f32[1536, 384, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_216: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_181, [1, 1536, 384]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_169: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_170: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    sum_57: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 2])
    sub_130: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170)
    mul_477: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_216, sub_130);  sub_130 = None
    sum_58: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 2]);  mul_477 = None
    mul_478: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_57, 0.0026041666666666665);  sum_57 = None
    unsqueeze_171: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
    unsqueeze_172: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    mul_479: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_58, 0.0026041666666666665)
    mul_480: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_481: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    unsqueeze_173: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_174: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    mul_482: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, view_127);  view_127 = None
    unsqueeze_175: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_176: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    sub_131: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170);  view_126 = unsqueeze_170 = None
    mul_483: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_174);  sub_131 = unsqueeze_174 = None
    sub_132: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_216, mul_483);  view_216 = mul_483 = None
    sub_133: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_172);  sub_132 = unsqueeze_172 = None
    mul_484: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_176);  sub_133 = unsqueeze_176 = None
    mul_485: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_85);  sum_58 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_217: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_485, [1536, 1, 1, 1]);  mul_485 = None
    mul_486: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_217, 0.09125009274634042);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_218: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_484, [1536, 384, 1, 1]);  mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_77: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_57)
    full_13: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_134: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_13, sigmoid_77);  full_13 = None
    mul_487: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, sub_134);  convolution_57 = sub_134 = None
    add_89: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_487, 1);  mul_487 = None
    mul_488: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_89);  sigmoid_77 = add_89 = None
    mul_489: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_180, mul_488);  getitem_180 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_59: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_489, mul_193, view_125, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_489 = mul_193 = view_125 = None
    getitem_183: "f32[8, 384, 14, 14]" = convolution_backward_23[0]
    getitem_184: "f32[384, 64, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_219: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_184, [1, 384, 576]);  getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_177: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_82, 0);  squeeze_82 = None
    unsqueeze_178: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    sum_60: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_219, [0, 2])
    sub_135: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178)
    mul_490: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_219, sub_135);  sub_135 = None
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2]);  mul_490 = None
    mul_491: "f32[384]" = torch.ops.aten.mul.Tensor(sum_60, 0.001736111111111111);  sum_60 = None
    unsqueeze_179: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_180: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    mul_492: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, 0.001736111111111111)
    mul_493: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_83, squeeze_83)
    mul_494: "f32[384]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_181: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_182: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    mul_495: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_83, view_124);  view_124 = None
    unsqueeze_183: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_184: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    sub_136: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178);  view_123 = unsqueeze_178 = None
    mul_496: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_182);  sub_136 = unsqueeze_182 = None
    sub_137: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_219, mul_496);  view_219 = mul_496 = None
    sub_138: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_180);  sub_137 = unsqueeze_180 = None
    mul_497: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_184);  sub_138 = unsqueeze_184 = None
    mul_498: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_83);  sum_61 = squeeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_220: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_498, [384, 1, 1, 1]);  mul_498 = None
    mul_499: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_220, 0.07450538873672485);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_221: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_497, [384, 64, 3, 3]);  mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_78: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_20)
    full_14: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_139: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_14, sigmoid_78);  full_14 = None
    mul_500: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_20, sub_139);  clone_20 = sub_139 = None
    add_90: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_500, 1);  mul_500 = None
    mul_501: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_90);  sigmoid_78 = add_90 = None
    mul_502: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_183, mul_501);  getitem_183 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_502, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_502, mul_189, view_122, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_502 = mul_189 = view_122 = None
    getitem_186: "f32[8, 384, 14, 14]" = convolution_backward_24[0]
    getitem_187: "f32[384, 64, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_222: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_187, [1, 384, 576]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_185: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_80, 0);  squeeze_80 = None
    unsqueeze_186: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_222, [0, 2])
    sub_140: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186)
    mul_503: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_222, sub_140);  sub_140 = None
    sum_64: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2]);  mul_503 = None
    mul_504: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, 0.001736111111111111);  sum_63 = None
    unsqueeze_187: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_188: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    mul_505: "f32[384]" = torch.ops.aten.mul.Tensor(sum_64, 0.001736111111111111)
    mul_506: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_81, squeeze_81)
    mul_507: "f32[384]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    unsqueeze_189: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_190: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    mul_508: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_81, view_121);  view_121 = None
    unsqueeze_191: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_192: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    sub_141: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186);  view_120 = unsqueeze_186 = None
    mul_509: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_190);  sub_141 = unsqueeze_190 = None
    sub_142: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_222, mul_509);  view_222 = mul_509 = None
    sub_143: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_188);  sub_142 = unsqueeze_188 = None
    mul_510: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_192);  sub_143 = unsqueeze_192 = None
    mul_511: "f32[384]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_81);  sum_64 = squeeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_223: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_511, [384, 1, 1, 1]);  mul_511 = None
    mul_512: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_223, 0.07450538873672485);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_224: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_510, [384, 64, 3, 3]);  mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_79: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_19)
    full_15: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_144: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_15, sigmoid_79);  full_15 = None
    mul_513: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_19, sub_144);  clone_19 = sub_144 = None
    add_91: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_513, 1);  mul_513 = None
    mul_514: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_91);  sigmoid_79 = add_91 = None
    mul_515: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_186, mul_514);  getitem_186 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_65: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_515, mul_185, view_119, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_515 = mul_185 = view_119 = None
    getitem_189: "f32[8, 1536, 14, 14]" = convolution_backward_25[0]
    getitem_190: "f32[384, 1536, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_225: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_190, [1, 384, 1536]);  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_193: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_194: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    sum_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 2])
    sub_145: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194)
    mul_516: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_225, sub_145);  sub_145 = None
    sum_67: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2]);  mul_516 = None
    mul_517: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006510416666666666);  sum_66 = None
    unsqueeze_195: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_196: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    mul_518: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006510416666666666)
    mul_519: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_520: "f32[384]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    unsqueeze_197: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_198: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    mul_521: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, view_118);  view_118 = None
    unsqueeze_199: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_200: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    sub_146: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194);  view_117 = unsqueeze_194 = None
    mul_522: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_198);  sub_146 = unsqueeze_198 = None
    sub_147: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_225, mul_522);  view_225 = mul_522 = None
    sub_148: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_196);  sub_147 = unsqueeze_196 = None
    mul_523: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_200);  sub_148 = unsqueeze_200 = None
    mul_524: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_79);  sum_67 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_226: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_524, [384, 1, 1, 1]);  mul_524 = None
    mul_525: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_226, 0.04562504637317021);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_227: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_523, [384, 1536, 1, 1]);  mul_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_526: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_189, 0.9128709291752768);  getitem_189 = None
    sigmoid_80: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_46)
    full_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.full.default([8, 1536, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_149: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_16, sigmoid_80);  full_16 = None
    mul_527: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, sub_149);  add_46 = sub_149 = None
    add_92: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_527, 1);  mul_527 = None
    mul_528: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_92);  sigmoid_80 = add_92 = None
    mul_529: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_526, mul_528);  mul_526 = mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_93: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_470, mul_529);  mul_470 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_530: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_93, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_531: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_530, 2.0);  mul_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_532: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_531, convolution_52);  convolution_52 = None
    mul_533: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_531, sigmoid_42);  mul_531 = sigmoid_42 = None
    sum_68: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [2, 3], True);  mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_40: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_150: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_40)
    mul_534: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_40, sub_150);  alias_40 = sub_150 = None
    mul_535: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_68, mul_534);  sum_68 = mul_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_69: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_535, relu_7, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = primals_202 = None
    getitem_192: "f32[8, 384, 1, 1]" = convolution_backward_26[0]
    getitem_193: "f32[1536, 384, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_42: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_43: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_4: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_192);  le_4 = scalar_tensor_4 = getitem_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_70: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_4, mean_7, primals_200, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = mean_7 = primals_200 = None
    getitem_195: "f32[8, 1536, 1, 1]" = convolution_backward_27[0]
    getitem_196: "f32[384, 1536, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_195, [8, 1536, 14, 14]);  getitem_195 = None
    div_5: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_94: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_533, div_5);  mul_533 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_71: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_94, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_94, mul_177, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_94 = mul_177 = view_116 = None
    getitem_198: "f32[8, 384, 14, 14]" = convolution_backward_28[0]
    getitem_199: "f32[1536, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_228: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_199, [1, 1536, 384]);  getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_201: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_76, 0);  squeeze_76 = None
    unsqueeze_202: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    sum_72: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_228, [0, 2])
    sub_151: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202)
    mul_536: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_228, sub_151);  sub_151 = None
    sum_73: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2]);  mul_536 = None
    mul_537: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_72, 0.0026041666666666665);  sum_72 = None
    unsqueeze_203: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_204: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    mul_538: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_73, 0.0026041666666666665)
    mul_539: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, squeeze_77)
    mul_540: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    unsqueeze_205: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_206: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    mul_541: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, view_115);  view_115 = None
    unsqueeze_207: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_208: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    sub_152: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202);  view_114 = unsqueeze_202 = None
    mul_542: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_206);  sub_152 = unsqueeze_206 = None
    sub_153: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_228, mul_542);  view_228 = mul_542 = None
    sub_154: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_204);  sub_153 = unsqueeze_204 = None
    mul_543: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_208);  sub_154 = unsqueeze_208 = None
    mul_544: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_77);  sum_73 = squeeze_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_229: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_544, [1536, 1, 1, 1]);  mul_544 = None
    mul_545: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_229, 0.09125009274634042);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_230: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_543, [1536, 384, 1, 1]);  mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_81: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_51)
    full_17: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_155: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_17, sigmoid_81);  full_17 = None
    mul_546: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, sub_155);  convolution_51 = sub_155 = None
    add_95: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_546, 1);  mul_546 = None
    mul_547: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_95);  sigmoid_81 = add_95 = None
    mul_548: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_198, mul_547);  getitem_198 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_74: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_548, mul_173, view_113, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_548 = mul_173 = view_113 = None
    getitem_201: "f32[8, 384, 14, 14]" = convolution_backward_29[0]
    getitem_202: "f32[384, 64, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_231: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_202, [1, 384, 576]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_209: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_74, 0);  squeeze_74 = None
    unsqueeze_210: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    sum_75: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_231, [0, 2])
    sub_156: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210)
    mul_549: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_231, sub_156);  sub_156 = None
    sum_76: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 2]);  mul_549 = None
    mul_550: "f32[384]" = torch.ops.aten.mul.Tensor(sum_75, 0.001736111111111111);  sum_75 = None
    unsqueeze_211: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
    unsqueeze_212: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    mul_551: "f32[384]" = torch.ops.aten.mul.Tensor(sum_76, 0.001736111111111111)
    mul_552: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_75, squeeze_75)
    mul_553: "f32[384]" = torch.ops.aten.mul.Tensor(mul_551, mul_552);  mul_551 = mul_552 = None
    unsqueeze_213: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_214: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    mul_554: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_75, view_112);  view_112 = None
    unsqueeze_215: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_216: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    sub_157: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210);  view_111 = unsqueeze_210 = None
    mul_555: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_214);  sub_157 = unsqueeze_214 = None
    sub_158: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_231, mul_555);  view_231 = mul_555 = None
    sub_159: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_212);  sub_158 = unsqueeze_212 = None
    mul_556: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_216);  sub_159 = unsqueeze_216 = None
    mul_557: "f32[384]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_75);  sum_76 = squeeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_232: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_557, [384, 1, 1, 1]);  mul_557 = None
    mul_558: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_232, 0.07450538873672485);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_233: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_556, [384, 64, 3, 3]);  mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_82: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_18)
    full_18: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_160: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_18, sigmoid_82);  full_18 = None
    mul_559: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_18, sub_160);  clone_18 = sub_160 = None
    add_96: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_559, 1);  mul_559 = None
    mul_560: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_96);  sigmoid_82 = add_96 = None
    mul_561: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_201, mul_560);  getitem_201 = mul_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_77: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_561, mul_169, view_110, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_561 = mul_169 = view_110 = None
    getitem_204: "f32[8, 384, 14, 14]" = convolution_backward_30[0]
    getitem_205: "f32[384, 64, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_234: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_205, [1, 384, 576]);  getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_217: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_218: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    sum_78: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_234, [0, 2])
    sub_161: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218)
    mul_562: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_234, sub_161);  sub_161 = None
    sum_79: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2]);  mul_562 = None
    mul_563: "f32[384]" = torch.ops.aten.mul.Tensor(sum_78, 0.001736111111111111);  sum_78 = None
    unsqueeze_219: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_220: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    mul_564: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, 0.001736111111111111)
    mul_565: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_566: "f32[384]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_221: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_222: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    mul_567: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_73, view_109);  view_109 = None
    unsqueeze_223: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_224: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    sub_162: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218);  view_108 = unsqueeze_218 = None
    mul_568: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_222);  sub_162 = unsqueeze_222 = None
    sub_163: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_234, mul_568);  view_234 = mul_568 = None
    sub_164: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_220);  sub_163 = unsqueeze_220 = None
    mul_569: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_224);  sub_164 = unsqueeze_224 = None
    mul_570: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_73);  sum_79 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_235: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_570, [384, 1, 1, 1]);  mul_570 = None
    mul_571: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_235, 0.07450538873672485);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_236: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_569, [384, 64, 3, 3]);  mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_83: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_17)
    full_19: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_165: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_19, sigmoid_83);  full_19 = None
    mul_572: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_17, sub_165);  clone_17 = sub_165 = None
    add_97: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_572, 1);  mul_572 = None
    mul_573: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_97);  sigmoid_83 = add_97 = None
    mul_574: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_204, mul_573);  getitem_204 = mul_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_80: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_574, mul_165, view_107, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = mul_165 = view_107 = None
    getitem_207: "f32[8, 1536, 14, 14]" = convolution_backward_31[0]
    getitem_208: "f32[384, 1536, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_237: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_208, [1, 384, 1536]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_225: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_70, 0);  squeeze_70 = None
    unsqueeze_226: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    sum_81: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_237, [0, 2])
    sub_166: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226)
    mul_575: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_237, sub_166);  sub_166 = None
    sum_82: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2]);  mul_575 = None
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006510416666666666);  sum_81 = None
    unsqueeze_227: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_228: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    mul_577: "f32[384]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006510416666666666)
    mul_578: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_71, squeeze_71)
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
    unsqueeze_229: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_230: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    mul_580: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_71, view_106);  view_106 = None
    unsqueeze_231: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_232: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    sub_167: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226);  view_105 = unsqueeze_226 = None
    mul_581: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_230);  sub_167 = unsqueeze_230 = None
    sub_168: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_237, mul_581);  view_237 = mul_581 = None
    sub_169: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_228);  sub_168 = unsqueeze_228 = None
    mul_582: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_232);  sub_169 = unsqueeze_232 = None
    mul_583: "f32[384]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_71);  sum_82 = squeeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_238: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_583, [384, 1, 1, 1]);  mul_583 = None
    mul_584: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_238, 0.04562504637317021);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_239: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_582, [384, 1536, 1, 1]);  mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_585: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_207, 0.9284766908852592);  getitem_207 = None
    sigmoid_84: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_41)
    full_20: "f32[8, 1536, 14, 14]" = torch.ops.aten.full.default([8, 1536, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_170: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_20, sigmoid_84);  full_20 = None
    mul_586: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sub_170);  add_41 = sub_170 = None
    add_98: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_586, 1);  mul_586 = None
    mul_587: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_98);  sigmoid_84 = add_98 = None
    mul_588: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_585, mul_587);  mul_585 = mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_99: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_93, mul_588);  add_93 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_589: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_99, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_590: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_589, 2.0);  mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_591: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_590, convolution_46);  convolution_46 = None
    mul_592: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_590, sigmoid_37);  mul_590 = sigmoid_37 = None
    sum_83: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_591, [2, 3], True);  mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_44: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    sub_171: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_44)
    mul_593: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_44, sub_171);  alias_44 = sub_171 = None
    mul_594: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, mul_593);  sum_83 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_84: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_594, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_594, relu_6, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_594 = primals_198 = None
    getitem_210: "f32[8, 384, 1, 1]" = convolution_backward_32[0]
    getitem_211: "f32[1536, 384, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_46: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_47: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_5: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_210);  le_5 = scalar_tensor_5 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_85: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_5, mean_6, primals_196, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_6 = primals_196 = None
    getitem_213: "f32[8, 1536, 1, 1]" = convolution_backward_33[0]
    getitem_214: "f32[384, 1536, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_213, [8, 1536, 14, 14]);  getitem_213 = None
    div_6: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_100: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_592, div_6);  mul_592 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_86: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_100, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(add_100, mul_157, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_100 = mul_157 = view_104 = None
    getitem_216: "f32[8, 384, 14, 14]" = convolution_backward_34[0]
    getitem_217: "f32[1536, 384, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_240: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_217, [1, 1536, 384]);  getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_233: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_68, 0);  squeeze_68 = None
    unsqueeze_234: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    sum_87: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_240, [0, 2])
    sub_172: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234)
    mul_595: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_240, sub_172);  sub_172 = None
    sum_88: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2]);  mul_595 = None
    mul_596: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_87, 0.0026041666666666665);  sum_87 = None
    unsqueeze_235: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_236: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    mul_597: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_88, 0.0026041666666666665)
    mul_598: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, squeeze_69)
    mul_599: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_597, mul_598);  mul_597 = mul_598 = None
    unsqueeze_237: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_238: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    mul_600: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, view_103);  view_103 = None
    unsqueeze_239: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_240: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    sub_173: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234);  view_102 = unsqueeze_234 = None
    mul_601: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_238);  sub_173 = unsqueeze_238 = None
    sub_174: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_240, mul_601);  view_240 = mul_601 = None
    sub_175: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_236);  sub_174 = unsqueeze_236 = None
    mul_602: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_240);  sub_175 = unsqueeze_240 = None
    mul_603: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_69);  sum_88 = squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_603, [1536, 1, 1, 1]);  mul_603 = None
    mul_604: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_241, 0.09125009274634042);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_242: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_602, [1536, 384, 1, 1]);  mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_85: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_45)
    full_21: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_176: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_21, sigmoid_85);  full_21 = None
    mul_605: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, sub_176);  convolution_45 = sub_176 = None
    add_101: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_605, 1);  mul_605 = None
    mul_606: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_101);  sigmoid_85 = add_101 = None
    mul_607: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_216, mul_606);  getitem_216 = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_89: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_607, mul_153, view_101, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_607 = mul_153 = view_101 = None
    getitem_219: "f32[8, 384, 14, 14]" = convolution_backward_35[0]
    getitem_220: "f32[384, 64, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_243: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_220, [1, 384, 576]);  getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_241: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_242: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    sum_90: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_243, [0, 2])
    sub_177: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242)
    mul_608: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_243, sub_177);  sub_177 = None
    sum_91: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2]);  mul_608 = None
    mul_609: "f32[384]" = torch.ops.aten.mul.Tensor(sum_90, 0.001736111111111111);  sum_90 = None
    unsqueeze_243: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_244: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    mul_610: "f32[384]" = torch.ops.aten.mul.Tensor(sum_91, 0.001736111111111111)
    mul_611: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_612: "f32[384]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    unsqueeze_245: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_246: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    mul_613: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_67, view_100);  view_100 = None
    unsqueeze_247: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_248: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    sub_178: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242);  view_99 = unsqueeze_242 = None
    mul_614: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_246);  sub_178 = unsqueeze_246 = None
    sub_179: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_243, mul_614);  view_243 = mul_614 = None
    sub_180: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_244);  sub_179 = unsqueeze_244 = None
    mul_615: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_248);  sub_180 = unsqueeze_248 = None
    mul_616: "f32[384]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_67);  sum_91 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_244: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_616, [384, 1, 1, 1]);  mul_616 = None
    mul_617: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_244, 0.07450538873672485);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_245: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_615, [384, 64, 3, 3]);  mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_86: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_16)
    full_22: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_181: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_22, sigmoid_86);  full_22 = None
    mul_618: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_16, sub_181);  clone_16 = sub_181 = None
    add_102: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_618, 1);  mul_618 = None
    mul_619: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_102);  sigmoid_86 = add_102 = None
    mul_620: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_219, mul_619);  getitem_219 = mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_92: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_620, mul_149, view_98, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_620 = mul_149 = view_98 = None
    getitem_222: "f32[8, 384, 14, 14]" = convolution_backward_36[0]
    getitem_223: "f32[384, 64, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_246: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_223, [1, 384, 576]);  getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_249: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_64, 0);  squeeze_64 = None
    unsqueeze_250: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    sum_93: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_246, [0, 2])
    sub_182: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250)
    mul_621: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_246, sub_182);  sub_182 = None
    sum_94: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 2]);  mul_621 = None
    mul_622: "f32[384]" = torch.ops.aten.mul.Tensor(sum_93, 0.001736111111111111);  sum_93 = None
    unsqueeze_251: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_252: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    mul_623: "f32[384]" = torch.ops.aten.mul.Tensor(sum_94, 0.001736111111111111)
    mul_624: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_65, squeeze_65)
    mul_625: "f32[384]" = torch.ops.aten.mul.Tensor(mul_623, mul_624);  mul_623 = mul_624 = None
    unsqueeze_253: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_254: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    mul_626: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_65, view_97);  view_97 = None
    unsqueeze_255: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    unsqueeze_256: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    sub_183: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250);  view_96 = unsqueeze_250 = None
    mul_627: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_254);  sub_183 = unsqueeze_254 = None
    sub_184: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_246, mul_627);  view_246 = mul_627 = None
    sub_185: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_252);  sub_184 = unsqueeze_252 = None
    mul_628: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_256);  sub_185 = unsqueeze_256 = None
    mul_629: "f32[384]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_65);  sum_94 = squeeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_247: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_629, [384, 1, 1, 1]);  mul_629 = None
    mul_630: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_247, 0.07450538873672485);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_248: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_628, [384, 64, 3, 3]);  mul_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_87: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_15)
    full_23: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_186: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_23, sigmoid_87);  full_23 = None
    mul_631: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_15, sub_186);  clone_15 = sub_186 = None
    add_103: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_631, 1);  mul_631 = None
    mul_632: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_103);  sigmoid_87 = add_103 = None
    mul_633: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_222, mul_632);  getitem_222 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_95: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_633, mul_145, view_95, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_633 = mul_145 = view_95 = None
    getitem_225: "f32[8, 1536, 14, 14]" = convolution_backward_37[0]
    getitem_226: "f32[384, 1536, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_249: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_226, [1, 384, 1536]);  getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_257: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_62, 0);  squeeze_62 = None
    unsqueeze_258: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    sum_96: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_249, [0, 2])
    sub_187: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258)
    mul_634: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_249, sub_187);  sub_187 = None
    sum_97: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2]);  mul_634 = None
    mul_635: "f32[384]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006510416666666666);  sum_96 = None
    unsqueeze_259: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_260: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    mul_636: "f32[384]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006510416666666666)
    mul_637: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_63, squeeze_63)
    mul_638: "f32[384]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_261: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_262: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    mul_639: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_63, view_94);  view_94 = None
    unsqueeze_263: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_264: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    sub_188: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258);  view_93 = unsqueeze_258 = None
    mul_640: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_262);  sub_188 = unsqueeze_262 = None
    sub_189: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_249, mul_640);  view_249 = mul_640 = None
    sub_190: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_260);  sub_189 = unsqueeze_260 = None
    mul_641: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_264);  sub_190 = unsqueeze_264 = None
    mul_642: "f32[384]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_63);  sum_97 = squeeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_250: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_642, [384, 1, 1, 1]);  mul_642 = None
    mul_643: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_250, 0.04562504637317021);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_251: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_641, [384, 1536, 1, 1]);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_644: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_225, 0.9449111825230679);  getitem_225 = None
    sigmoid_88: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_36)
    full_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.full.default([8, 1536, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_191: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_24, sigmoid_88);  full_24 = None
    mul_645: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_36, sub_191);  add_36 = sub_191 = None
    add_104: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_645, 1);  mul_645 = None
    mul_646: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_104);  sigmoid_88 = add_104 = None
    mul_647: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_644, mul_646);  mul_644 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_105: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_99, mul_647);  add_99 = mul_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_648: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_105, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_649: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_648, 2.0);  mul_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_650: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_649, convolution_40);  convolution_40 = None
    mul_651: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_649, sigmoid_32);  mul_649 = sigmoid_32 = None
    sum_98: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_650, [2, 3], True);  mul_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_48: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_192: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_48)
    mul_652: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_48, sub_192);  alias_48 = sub_192 = None
    mul_653: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_98, mul_652);  sum_98 = mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_99: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_653, relu_5, primals_194, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_653 = primals_194 = None
    getitem_228: "f32[8, 384, 1, 1]" = convolution_backward_38[0]
    getitem_229: "f32[1536, 384, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_50: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_51: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_6: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_228);  le_6 = scalar_tensor_6 = getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_100: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_6, mean_5, primals_192, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_5 = primals_192 = None
    getitem_231: "f32[8, 1536, 1, 1]" = convolution_backward_39[0]
    getitem_232: "f32[384, 1536, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_231, [8, 1536, 14, 14]);  getitem_231 = None
    div_7: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_106: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_651, div_7);  mul_651 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_101: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_106, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_106, mul_137, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_106 = mul_137 = view_92 = None
    getitem_234: "f32[8, 384, 14, 14]" = convolution_backward_40[0]
    getitem_235: "f32[1536, 384, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_252: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_235, [1, 1536, 384]);  getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_265: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_266: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    sum_102: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_252, [0, 2])
    sub_193: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266)
    mul_654: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_252, sub_193);  sub_193 = None
    sum_103: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 2]);  mul_654 = None
    mul_655: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_102, 0.0026041666666666665);  sum_102 = None
    unsqueeze_267: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_268: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    mul_656: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_103, 0.0026041666666666665)
    mul_657: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_658: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_656, mul_657);  mul_656 = mul_657 = None
    unsqueeze_269: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_270: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    mul_659: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, view_91);  view_91 = None
    unsqueeze_271: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
    unsqueeze_272: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    sub_194: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266);  view_90 = unsqueeze_266 = None
    mul_660: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_270);  sub_194 = unsqueeze_270 = None
    sub_195: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_252, mul_660);  view_252 = mul_660 = None
    sub_196: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_268);  sub_195 = unsqueeze_268 = None
    mul_661: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_272);  sub_196 = unsqueeze_272 = None
    mul_662: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_61);  sum_103 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_253: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_662, [1536, 1, 1, 1]);  mul_662 = None
    mul_663: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_253, 0.09125009274634042);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_254: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_661, [1536, 384, 1, 1]);  mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_89: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_39)
    full_25: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_197: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_25, sigmoid_89);  full_25 = None
    mul_664: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, sub_197);  convolution_39 = sub_197 = None
    add_107: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_664, 1);  mul_664 = None
    mul_665: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_107);  sigmoid_89 = add_107 = None
    mul_666: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_234, mul_665);  getitem_234 = mul_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_104: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_666, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_666, mul_133, view_89, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_666 = mul_133 = view_89 = None
    getitem_237: "f32[8, 384, 14, 14]" = convolution_backward_41[0]
    getitem_238: "f32[384, 64, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_255: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_238, [1, 384, 576]);  getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_273: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_58, 0);  squeeze_58 = None
    unsqueeze_274: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    sum_105: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_255, [0, 2])
    sub_198: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274)
    mul_667: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_255, sub_198);  sub_198 = None
    sum_106: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2]);  mul_667 = None
    mul_668: "f32[384]" = torch.ops.aten.mul.Tensor(sum_105, 0.001736111111111111);  sum_105 = None
    unsqueeze_275: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_276: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    mul_669: "f32[384]" = torch.ops.aten.mul.Tensor(sum_106, 0.001736111111111111)
    mul_670: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_59, squeeze_59)
    mul_671: "f32[384]" = torch.ops.aten.mul.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_277: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_278: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    mul_672: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_59, view_88);  view_88 = None
    unsqueeze_279: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_280: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    sub_199: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274);  view_87 = unsqueeze_274 = None
    mul_673: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_278);  sub_199 = unsqueeze_278 = None
    sub_200: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_255, mul_673);  view_255 = mul_673 = None
    sub_201: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_276);  sub_200 = unsqueeze_276 = None
    mul_674: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_280);  sub_201 = unsqueeze_280 = None
    mul_675: "f32[384]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_59);  sum_106 = squeeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_256: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_675, [384, 1, 1, 1]);  mul_675 = None
    mul_676: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_256, 0.07450538873672485);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_257: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_674, [384, 64, 3, 3]);  mul_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_90: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_14)
    full_26: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_202: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_26, sigmoid_90);  full_26 = None
    mul_677: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_14, sub_202);  clone_14 = sub_202 = None
    add_108: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_677, 1);  mul_677 = None
    mul_678: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_108);  sigmoid_90 = add_108 = None
    mul_679: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_237, mul_678);  getitem_237 = mul_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_107: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_679, mul_129, view_86, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_679 = mul_129 = view_86 = None
    getitem_240: "f32[8, 384, 14, 14]" = convolution_backward_42[0]
    getitem_241: "f32[384, 64, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_258: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_241, [1, 384, 576]);  getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_281: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_56, 0);  squeeze_56 = None
    unsqueeze_282: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    sum_108: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_258, [0, 2])
    sub_203: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282)
    mul_680: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_258, sub_203);  sub_203 = None
    sum_109: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2]);  mul_680 = None
    mul_681: "f32[384]" = torch.ops.aten.mul.Tensor(sum_108, 0.001736111111111111);  sum_108 = None
    unsqueeze_283: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_284: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    mul_682: "f32[384]" = torch.ops.aten.mul.Tensor(sum_109, 0.001736111111111111)
    mul_683: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_57, squeeze_57)
    mul_684: "f32[384]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_285: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_286: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    mul_685: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_57, view_85);  view_85 = None
    unsqueeze_287: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_288: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    sub_204: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282);  view_84 = unsqueeze_282 = None
    mul_686: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_286);  sub_204 = unsqueeze_286 = None
    sub_205: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_258, mul_686);  view_258 = mul_686 = None
    sub_206: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_284);  sub_205 = unsqueeze_284 = None
    mul_687: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_288);  sub_206 = unsqueeze_288 = None
    mul_688: "f32[384]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_57);  sum_109 = squeeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_259: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_688, [384, 1, 1, 1]);  mul_688 = None
    mul_689: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_259, 0.07450538873672485);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_260: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_687, [384, 64, 3, 3]);  mul_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_91: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_13)
    full_27: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_207: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_27, sigmoid_91);  full_27 = None
    mul_690: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_13, sub_207);  clone_13 = sub_207 = None
    add_109: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_690, 1);  mul_690 = None
    mul_691: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_109);  sigmoid_91 = add_109 = None
    mul_692: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_240, mul_691);  getitem_240 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_110: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_692, mul_125, view_83, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_692 = mul_125 = view_83 = None
    getitem_243: "f32[8, 1536, 14, 14]" = convolution_backward_43[0]
    getitem_244: "f32[384, 1536, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_261: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_244, [1, 384, 1536]);  getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_289: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_290: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    sum_111: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_261, [0, 2])
    sub_208: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290)
    mul_693: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_261, sub_208);  sub_208 = None
    sum_112: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 2]);  mul_693 = None
    mul_694: "f32[384]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006510416666666666);  sum_111 = None
    unsqueeze_291: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_292: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    mul_695: "f32[384]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006510416666666666)
    mul_696: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_697: "f32[384]" = torch.ops.aten.mul.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
    unsqueeze_293: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
    unsqueeze_294: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    mul_698: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_55, view_82);  view_82 = None
    unsqueeze_295: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_296: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    sub_209: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290);  view_81 = unsqueeze_290 = None
    mul_699: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_294);  sub_209 = unsqueeze_294 = None
    sub_210: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_261, mul_699);  view_261 = mul_699 = None
    sub_211: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_292);  sub_210 = unsqueeze_292 = None
    mul_700: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_296);  sub_211 = unsqueeze_296 = None
    mul_701: "f32[384]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_55);  sum_112 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_262: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_701, [384, 1, 1, 1]);  mul_701 = None
    mul_702: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_262, 0.04562504637317021);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_263: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_700, [384, 1536, 1, 1]);  mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_703: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_243, 0.9622504486493761);  getitem_243 = None
    sigmoid_92: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_31)
    full_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.full.default([8, 1536, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_212: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_28, sigmoid_92);  full_28 = None
    mul_704: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_31, sub_212);  add_31 = sub_212 = None
    add_110: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_704, 1);  mul_704 = None
    mul_705: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_110);  sigmoid_92 = add_110 = None
    mul_706: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_703, mul_705);  mul_703 = mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_111: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_105, mul_706);  add_105 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_707: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_111, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_708: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_707, 2.0);  mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_709: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_708, convolution_34);  convolution_34 = None
    mul_710: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_708, sigmoid_27);  mul_708 = sigmoid_27 = None
    sum_113: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_709, [2, 3], True);  mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_52: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_213: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_52)
    mul_711: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_52, sub_213);  alias_52 = sub_213 = None
    mul_712: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_113, mul_711);  sum_113 = mul_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_114: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_712, relu_4, primals_190, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_712 = primals_190 = None
    getitem_246: "f32[8, 384, 1, 1]" = convolution_backward_44[0]
    getitem_247: "f32[1536, 384, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_54: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_55: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_7: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_246);  le_7 = scalar_tensor_7 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_115: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_7, mean_4, primals_188, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = mean_4 = primals_188 = None
    getitem_249: "f32[8, 1536, 1, 1]" = convolution_backward_45[0]
    getitem_250: "f32[384, 1536, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_249, [8, 1536, 14, 14]);  getitem_249 = None
    div_8: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_112: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_710, div_8);  mul_710 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_116: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_112, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_112, mul_117, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_112 = mul_117 = view_80 = None
    getitem_252: "f32[8, 384, 14, 14]" = convolution_backward_46[0]
    getitem_253: "f32[1536, 384, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_264: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_253, [1, 1536, 384]);  getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_297: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_52, 0);  squeeze_52 = None
    unsqueeze_298: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    sum_117: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_264, [0, 2])
    sub_214: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298)
    mul_713: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_264, sub_214);  sub_214 = None
    sum_118: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 2]);  mul_713 = None
    mul_714: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_117, 0.0026041666666666665);  sum_117 = None
    unsqueeze_299: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
    unsqueeze_300: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    mul_715: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_118, 0.0026041666666666665)
    mul_716: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, squeeze_53)
    mul_717: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
    unsqueeze_301: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_302: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    mul_718: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, view_79);  view_79 = None
    unsqueeze_303: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_304: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    sub_215: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298);  view_78 = unsqueeze_298 = None
    mul_719: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_302);  sub_215 = unsqueeze_302 = None
    sub_216: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_264, mul_719);  view_264 = mul_719 = None
    sub_217: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_300);  sub_216 = unsqueeze_300 = None
    mul_720: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_304);  sub_217 = unsqueeze_304 = None
    mul_721: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_53);  sum_118 = squeeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_265: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_721, [1536, 1, 1, 1]);  mul_721 = None
    mul_722: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_265, 0.09125009274634042);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_266: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_720, [1536, 384, 1, 1]);  mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_93: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_33)
    full_29: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_218: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_29, sigmoid_93);  full_29 = None
    mul_723: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, sub_218);  convolution_33 = sub_218 = None
    add_113: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_723, 1);  mul_723 = None
    mul_724: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_113);  sigmoid_93 = add_113 = None
    mul_725: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_252, mul_724);  getitem_252 = mul_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_119: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_725, mul_113, view_77, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_725 = mul_113 = view_77 = None
    getitem_255: "f32[8, 384, 14, 14]" = convolution_backward_47[0]
    getitem_256: "f32[384, 64, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_267: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_256, [1, 384, 576]);  getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_305: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_50, 0);  squeeze_50 = None
    unsqueeze_306: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    sum_120: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_267, [0, 2])
    sub_219: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306)
    mul_726: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_267, sub_219);  sub_219 = None
    sum_121: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 2]);  mul_726 = None
    mul_727: "f32[384]" = torch.ops.aten.mul.Tensor(sum_120, 0.001736111111111111);  sum_120 = None
    unsqueeze_307: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_308: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    mul_728: "f32[384]" = torch.ops.aten.mul.Tensor(sum_121, 0.001736111111111111)
    mul_729: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_51, squeeze_51)
    mul_730: "f32[384]" = torch.ops.aten.mul.Tensor(mul_728, mul_729);  mul_728 = mul_729 = None
    unsqueeze_309: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_310: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    mul_731: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_51, view_76);  view_76 = None
    unsqueeze_311: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_312: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    sub_220: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306);  view_75 = unsqueeze_306 = None
    mul_732: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_310);  sub_220 = unsqueeze_310 = None
    sub_221: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_267, mul_732);  view_267 = mul_732 = None
    sub_222: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_308);  sub_221 = unsqueeze_308 = None
    mul_733: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_312);  sub_222 = unsqueeze_312 = None
    mul_734: "f32[384]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_51);  sum_121 = squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_268: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_734, [384, 1, 1, 1]);  mul_734 = None
    mul_735: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_268, 0.07450538873672485);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_269: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_733, [384, 64, 3, 3]);  mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_94: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_12)
    full_30: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_223: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_30, sigmoid_94);  full_30 = None
    mul_736: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_12, sub_223);  clone_12 = sub_223 = None
    add_114: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_736, 1);  mul_736 = None
    mul_737: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_114);  sigmoid_94 = add_114 = None
    mul_738: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_255, mul_737);  getitem_255 = mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_122: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_738, mul_109, view_74, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_738 = mul_109 = view_74 = None
    getitem_258: "f32[8, 384, 14, 14]" = convolution_backward_48[0]
    getitem_259: "f32[384, 64, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_270: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_259, [1, 384, 576]);  getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_313: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_314: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    sum_123: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 2])
    sub_224: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314)
    mul_739: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_270, sub_224);  sub_224 = None
    sum_124: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2]);  mul_739 = None
    mul_740: "f32[384]" = torch.ops.aten.mul.Tensor(sum_123, 0.001736111111111111);  sum_123 = None
    unsqueeze_315: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_316: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    mul_741: "f32[384]" = torch.ops.aten.mul.Tensor(sum_124, 0.001736111111111111)
    mul_742: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_743: "f32[384]" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    unsqueeze_317: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_318: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    mul_744: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, view_73);  view_73 = None
    unsqueeze_319: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_320: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    sub_225: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314);  view_72 = unsqueeze_314 = None
    mul_745: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_318);  sub_225 = unsqueeze_318 = None
    sub_226: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_270, mul_745);  view_270 = mul_745 = None
    sub_227: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_316);  sub_226 = unsqueeze_316 = None
    mul_746: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_320);  sub_227 = unsqueeze_320 = None
    mul_747: "f32[384]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_49);  sum_124 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_271: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_747, [384, 1, 1, 1]);  mul_747 = None
    mul_748: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_271, 0.07450538873672485);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_272: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_746, [384, 64, 3, 3]);  mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_95: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_11)
    full_31: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_228: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_31, sigmoid_95);  full_31 = None
    mul_749: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_11, sub_228);  clone_11 = sub_228 = None
    add_115: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_749, 1);  mul_749 = None
    mul_750: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_115);  sigmoid_95 = add_115 = None
    mul_751: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_258, mul_750);  getitem_258 = mul_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_125: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_751, mul_105, view_71, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_751 = mul_105 = view_71 = None
    getitem_261: "f32[8, 1536, 14, 14]" = convolution_backward_49[0]
    getitem_262: "f32[384, 1536, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_273: "f32[1, 384, 1536]" = torch.ops.aten.view.default(getitem_262, [1, 384, 1536]);  getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_321: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_46, 0);  squeeze_46 = None
    unsqueeze_322: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    sum_126: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 2])
    sub_229: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322)
    mul_752: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_273, sub_229);  sub_229 = None
    sum_127: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2]);  mul_752 = None
    mul_753: "f32[384]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006510416666666666);  sum_126 = None
    unsqueeze_323: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_324: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    mul_754: "f32[384]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006510416666666666)
    mul_755: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_47, squeeze_47)
    mul_756: "f32[384]" = torch.ops.aten.mul.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
    unsqueeze_325: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_326: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    mul_757: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_47, view_70);  view_70 = None
    unsqueeze_327: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_328: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    sub_230: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322);  view_69 = unsqueeze_322 = None
    mul_758: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_326);  sub_230 = unsqueeze_326 = None
    sub_231: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_273, mul_758);  view_273 = mul_758 = None
    sub_232: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_324);  sub_231 = unsqueeze_324 = None
    mul_759: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_328);  sub_232 = unsqueeze_328 = None
    mul_760: "f32[384]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_47);  sum_127 = squeeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_274: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_760, [384, 1, 1, 1]);  mul_760 = None
    mul_761: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_274, 0.04562504637317021);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_275: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_759, [384, 1536, 1, 1]);  mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_762: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_261, 0.9805806756909201);  getitem_261 = None
    sigmoid_96: "f32[8, 1536, 14, 14]" = torch.ops.aten.sigmoid.default(add_26)
    full_32: "f32[8, 1536, 14, 14]" = torch.ops.aten.full.default([8, 1536, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_233: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(full_32, sigmoid_96);  full_32 = None
    mul_763: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_26, sub_233);  add_26 = sub_233 = None
    add_116: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Scalar(mul_763, 1);  mul_763 = None
    mul_764: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_96, add_116);  sigmoid_96 = add_116 = None
    mul_765: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_762, mul_764);  mul_762 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_117: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_111, mul_765);  add_111 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_766: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_117, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_767: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_766, 2.0);  mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_768: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_767, convolution_28);  convolution_28 = None
    mul_769: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_767, sigmoid_22);  mul_767 = sigmoid_22 = None
    sum_128: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2, 3], True);  mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_56: "f32[8, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_234: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_56)
    mul_770: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_56, sub_234);  alias_56 = sub_234 = None
    mul_771: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_128, mul_770);  sum_128 = mul_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_129: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_771, relu_3, primals_186, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_771 = primals_186 = None
    getitem_264: "f32[8, 384, 1, 1]" = convolution_backward_50[0]
    getitem_265: "f32[1536, 384, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_58: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_59: "f32[8, 384, 1, 1]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_8: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_264);  le_8 = scalar_tensor_8 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_130: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_8, mean_3, primals_184, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = mean_3 = primals_184 = None
    getitem_267: "f32[8, 1536, 1, 1]" = convolution_backward_51[0]
    getitem_268: "f32[384, 1536, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_267, [8, 1536, 14, 14]);  getitem_267 = None
    div_9: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_118: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_769, div_9);  mul_769 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_131: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_118, mul_97, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_118 = mul_97 = view_68 = None
    getitem_270: "f32[8, 384, 14, 14]" = convolution_backward_52[0]
    getitem_271: "f32[1536, 384, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_276: "f32[1, 1536, 384]" = torch.ops.aten.view.default(getitem_271, [1, 1536, 384]);  getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_329: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_44, 0);  squeeze_44 = None
    unsqueeze_330: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    sum_132: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 2])
    sub_235: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330)
    mul_772: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_276, sub_235);  sub_235 = None
    sum_133: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2]);  mul_772 = None
    mul_773: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_132, 0.0026041666666666665);  sum_132 = None
    unsqueeze_331: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_332: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    mul_774: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_133, 0.0026041666666666665)
    mul_775: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, squeeze_45)
    mul_776: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_333: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_334: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    mul_777: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, view_67);  view_67 = None
    unsqueeze_335: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_336: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    sub_236: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330);  view_66 = unsqueeze_330 = None
    mul_778: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_334);  sub_236 = unsqueeze_334 = None
    sub_237: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_276, mul_778);  view_276 = mul_778 = None
    sub_238: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_332);  sub_237 = unsqueeze_332 = None
    mul_779: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_336);  sub_238 = unsqueeze_336 = None
    mul_780: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_45);  sum_133 = squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_277: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_780, [1536, 1, 1, 1]);  mul_780 = None
    mul_781: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_277, 0.09125009274634042);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_278: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_779, [1536, 384, 1, 1]);  mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_97: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_27)
    full_33: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_239: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_33, sigmoid_97);  full_33 = None
    mul_782: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, sub_239);  convolution_27 = sub_239 = None
    add_119: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_782, 1);  mul_782 = None
    mul_783: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_97, add_119);  sigmoid_97 = add_119 = None
    mul_784: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_270, mul_783);  getitem_270 = mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_134: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_784, mul_93, view_65, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_784 = mul_93 = view_65 = None
    getitem_273: "f32[8, 384, 14, 14]" = convolution_backward_53[0]
    getitem_274: "f32[384, 64, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_279: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_274, [1, 384, 576]);  getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_337: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_338: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    sum_135: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_279, [0, 2])
    sub_240: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338)
    mul_785: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_279, sub_240);  sub_240 = None
    sum_136: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_785, [0, 2]);  mul_785 = None
    mul_786: "f32[384]" = torch.ops.aten.mul.Tensor(sum_135, 0.001736111111111111);  sum_135 = None
    unsqueeze_339: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_340: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    mul_787: "f32[384]" = torch.ops.aten.mul.Tensor(sum_136, 0.001736111111111111)
    mul_788: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_789: "f32[384]" = torch.ops.aten.mul.Tensor(mul_787, mul_788);  mul_787 = mul_788 = None
    unsqueeze_341: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    unsqueeze_342: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    mul_790: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, view_64);  view_64 = None
    unsqueeze_343: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    unsqueeze_344: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    sub_241: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338);  view_63 = unsqueeze_338 = None
    mul_791: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_342);  sub_241 = unsqueeze_342 = None
    sub_242: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_279, mul_791);  view_279 = mul_791 = None
    sub_243: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_340);  sub_242 = unsqueeze_340 = None
    mul_792: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_344);  sub_243 = unsqueeze_344 = None
    mul_793: "f32[384]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_43);  sum_136 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_280: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_793, [384, 1, 1, 1]);  mul_793 = None
    mul_794: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_280, 0.07450538873672485);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_281: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_792, [384, 64, 3, 3]);  mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_98: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(clone_10)
    full_34: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_244: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_34, sigmoid_98);  full_34 = None
    mul_795: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_10, sub_244);  clone_10 = sub_244 = None
    add_120: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_795, 1);  mul_795 = None
    mul_796: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_120);  sigmoid_98 = add_120 = None
    mul_797: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_273, mul_796);  getitem_273 = mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_137: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3])
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_797, mul_89, view_62, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_797 = mul_89 = view_62 = None
    getitem_276: "f32[8, 384, 28, 28]" = convolution_backward_54[0]
    getitem_277: "f32[384, 64, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_282: "f32[1, 384, 576]" = torch.ops.aten.view.default(getitem_277, [1, 384, 576]);  getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_345: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_40, 0);  squeeze_40 = None
    unsqueeze_346: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    sum_138: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 2])
    sub_245: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346)
    mul_798: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_282, sub_245);  sub_245 = None
    sum_139: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2]);  mul_798 = None
    mul_799: "f32[384]" = torch.ops.aten.mul.Tensor(sum_138, 0.001736111111111111);  sum_138 = None
    unsqueeze_347: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_348: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    mul_800: "f32[384]" = torch.ops.aten.mul.Tensor(sum_139, 0.001736111111111111)
    mul_801: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_41, squeeze_41)
    mul_802: "f32[384]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
    unsqueeze_349: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_350: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    mul_803: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_41, view_61);  view_61 = None
    unsqueeze_351: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_352: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    sub_246: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346);  view_60 = unsqueeze_346 = None
    mul_804: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_350);  sub_246 = unsqueeze_350 = None
    sub_247: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_282, mul_804);  view_282 = mul_804 = None
    sub_248: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_348);  sub_247 = unsqueeze_348 = None
    mul_805: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_352);  sub_248 = unsqueeze_352 = None
    mul_806: "f32[384]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_41);  sum_139 = squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_283: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_806, [384, 1, 1, 1]);  mul_806 = None
    mul_807: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_283, 0.07450538873672485);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_284: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_805, [384, 64, 3, 3]);  mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_99: "f32[8, 384, 28, 28]" = torch.ops.aten.sigmoid.default(clone_9)
    full_35: "f32[8, 384, 28, 28]" = torch.ops.aten.full.default([8, 384, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_249: "f32[8, 384, 28, 28]" = torch.ops.aten.sub.Tensor(full_35, sigmoid_99);  full_35 = None
    mul_808: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(clone_9, sub_249);  clone_9 = sub_249 = None
    add_121: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Scalar(mul_808, 1);  mul_808 = None
    mul_809: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_121);  sigmoid_99 = add_121 = None
    mul_810: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_276, mul_809);  getitem_276 = mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_140: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_810, [0, 2, 3])
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_810, mul_82, view_59, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_810 = view_59 = None
    getitem_279: "f32[8, 512, 28, 28]" = convolution_backward_55[0]
    getitem_280: "f32[384, 512, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_285: "f32[1, 384, 512]" = torch.ops.aten.view.default(getitem_280, [1, 384, 512]);  getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_353: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_38, 0);  squeeze_38 = None
    unsqueeze_354: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    sum_141: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_285, [0, 2])
    sub_250: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354)
    mul_811: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(view_285, sub_250);  sub_250 = None
    sum_142: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2]);  mul_811 = None
    mul_812: "f32[384]" = torch.ops.aten.mul.Tensor(sum_141, 0.001953125);  sum_141 = None
    unsqueeze_355: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_356: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    mul_813: "f32[384]" = torch.ops.aten.mul.Tensor(sum_142, 0.001953125)
    mul_814: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_39, squeeze_39)
    mul_815: "f32[384]" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    unsqueeze_357: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_358: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    mul_816: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_39, view_58);  view_58 = None
    unsqueeze_359: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_360: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    sub_251: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354);  view_57 = unsqueeze_354 = None
    mul_817: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_358);  sub_251 = unsqueeze_358 = None
    sub_252: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_285, mul_817);  view_285 = mul_817 = None
    sub_253: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_356);  sub_252 = unsqueeze_356 = None
    mul_818: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_360);  sub_253 = unsqueeze_360 = None
    mul_819: "f32[384]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_39);  sum_142 = squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_286: "f32[384, 1, 1, 1]" = torch.ops.aten.view.default(mul_819, [384, 1, 1, 1]);  mul_819 = None
    mul_820: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_286, 0.07902489841601695);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_287: "f32[384, 512, 1, 1]" = torch.ops.aten.view.default(mul_818, [384, 512, 1, 1]);  mul_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_143: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 2, 3])
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(add_117, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_117 = avg_pool2d_1 = view_56 = None
    getitem_282: "f32[8, 512, 14, 14]" = convolution_backward_56[0]
    getitem_283: "f32[1536, 512, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_288: "f32[1, 1536, 512]" = torch.ops.aten.view.default(getitem_283, [1, 1536, 512]);  getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_361: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_362: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    sum_144: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_288, [0, 2])
    sub_254: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362)
    mul_821: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(view_288, sub_254);  sub_254 = None
    sum_145: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_821, [0, 2]);  mul_821 = None
    mul_822: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_144, 0.001953125);  sum_144 = None
    unsqueeze_363: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_364: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    mul_823: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_145, 0.001953125)
    mul_824: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_825: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
    unsqueeze_365: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_366: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    mul_826: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, view_55);  view_55 = None
    unsqueeze_367: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
    unsqueeze_368: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    sub_255: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362);  view_54 = unsqueeze_362 = None
    mul_827: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_366);  sub_255 = unsqueeze_366 = None
    sub_256: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_288, mul_827);  view_288 = mul_827 = None
    sub_257: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_364);  sub_256 = unsqueeze_364 = None
    mul_828: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_368);  sub_257 = unsqueeze_368 = None
    mul_829: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_37);  sum_145 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_289: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_829, [1536, 1, 1, 1]);  mul_829 = None
    mul_830: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_289, 0.07902489841601695);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_290: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_828, [1536, 512, 1, 1]);  mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_1: "f32[8, 512, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(getitem_282, mul_82, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_282 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_122: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_279, avg_pool2d_backward_1);  getitem_279 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_831: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_122, 0.9622504486493761);  add_122 = None
    sigmoid_100: "f32[8, 512, 28, 28]" = torch.ops.aten.sigmoid.default(add_20)
    full_36: "f32[8, 512, 28, 28]" = torch.ops.aten.full.default([8, 512, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_258: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(full_36, sigmoid_100);  full_36 = None
    mul_832: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_20, sub_258);  add_20 = sub_258 = None
    add_123: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Scalar(mul_832, 1);  mul_832 = None
    mul_833: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_100, add_123);  sigmoid_100 = add_123 = None
    mul_834: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_831, mul_833);  mul_831 = mul_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_835: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_834, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_836: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_835, 2.0);  mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_837: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_836, convolution_21);  convolution_21 = None
    mul_838: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_836, sigmoid_17);  mul_836 = sigmoid_17 = None
    sum_146: "f32[8, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [2, 3], True);  mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_60: "f32[8, 512, 1, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    sub_259: "f32[8, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_60)
    mul_839: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(alias_60, sub_259);  alias_60 = sub_259 = None
    mul_840: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_146, mul_839);  sum_146 = mul_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_147: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 2, 3])
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_840, relu_2, primals_182, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_840 = primals_182 = None
    getitem_285: "f32[8, 128, 1, 1]" = convolution_backward_57[0]
    getitem_286: "f32[512, 128, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_62: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_63: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_9: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_285);  le_9 = scalar_tensor_9 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_148: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_9, mean_2, primals_180, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_2 = primals_180 = None
    getitem_288: "f32[8, 512, 1, 1]" = convolution_backward_58[0]
    getitem_289: "f32[128, 512, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 512, 28, 28]" = torch.ops.aten.expand.default(getitem_288, [8, 512, 28, 28]);  getitem_288 = None
    div_10: "f32[8, 512, 28, 28]" = torch.ops.aten.div.Scalar(expand_10, 784);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_124: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_838, div_10);  mul_838 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_149: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_124, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_124, mul_74, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_124 = mul_74 = view_53 = None
    getitem_291: "f32[8, 128, 28, 28]" = convolution_backward_59[0]
    getitem_292: "f32[512, 128, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_291: "f32[1, 512, 128]" = torch.ops.aten.view.default(getitem_292, [1, 512, 128]);  getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_369: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_34, 0);  squeeze_34 = None
    unsqueeze_370: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    sum_150: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_291, [0, 2])
    sub_260: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370)
    mul_841: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_291, sub_260);  sub_260 = None
    sum_151: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_841, [0, 2]);  mul_841 = None
    mul_842: "f32[512]" = torch.ops.aten.mul.Tensor(sum_150, 0.0078125);  sum_150 = None
    unsqueeze_371: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_372: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    mul_843: "f32[512]" = torch.ops.aten.mul.Tensor(sum_151, 0.0078125)
    mul_844: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, squeeze_35)
    mul_845: "f32[512]" = torch.ops.aten.mul.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
    unsqueeze_373: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
    unsqueeze_374: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    mul_846: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, view_52);  view_52 = None
    unsqueeze_375: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_376: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    sub_261: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370);  view_51 = unsqueeze_370 = None
    mul_847: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_374);  sub_261 = unsqueeze_374 = None
    sub_262: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_291, mul_847);  view_291 = mul_847 = None
    sub_263: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_372);  sub_262 = unsqueeze_372 = None
    mul_848: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_376);  sub_263 = unsqueeze_376 = None
    mul_849: "f32[512]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_35);  sum_151 = squeeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_292: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_849, [512, 1, 1, 1]);  mul_849 = None
    mul_850: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_292, 0.1580497968320339);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_293: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_848, [512, 128, 1, 1]);  mul_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_101: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_20)
    full_37: "f32[8, 128, 28, 28]" = torch.ops.aten.full.default([8, 128, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_264: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_37, sigmoid_101);  full_37 = None
    mul_851: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, sub_264);  convolution_20 = sub_264 = None
    add_125: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_851, 1);  mul_851 = None
    mul_852: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_101, add_125);  sigmoid_101 = add_125 = None
    mul_853: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_291, mul_852);  getitem_291 = mul_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_152: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2, 3])
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_853, mul_70, view_50, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_853 = mul_70 = view_50 = None
    getitem_294: "f32[8, 128, 28, 28]" = convolution_backward_60[0]
    getitem_295: "f32[128, 64, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_294: "f32[1, 128, 576]" = torch.ops.aten.view.default(getitem_295, [1, 128, 576]);  getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_377: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_32, 0);  squeeze_32 = None
    unsqueeze_378: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    sum_153: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_294, [0, 2])
    sub_265: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378)
    mul_854: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_294, sub_265);  sub_265 = None
    sum_154: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_854, [0, 2]);  mul_854 = None
    mul_855: "f32[128]" = torch.ops.aten.mul.Tensor(sum_153, 0.001736111111111111);  sum_153 = None
    unsqueeze_379: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_380: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    mul_856: "f32[128]" = torch.ops.aten.mul.Tensor(sum_154, 0.001736111111111111)
    mul_857: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, squeeze_33)
    mul_858: "f32[128]" = torch.ops.aten.mul.Tensor(mul_856, mul_857);  mul_856 = mul_857 = None
    unsqueeze_381: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_382: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    mul_859: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, view_49);  view_49 = None
    unsqueeze_383: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_384: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    sub_266: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378);  view_48 = unsqueeze_378 = None
    mul_860: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_382);  sub_266 = unsqueeze_382 = None
    sub_267: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_294, mul_860);  view_294 = mul_860 = None
    sub_268: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_380);  sub_267 = unsqueeze_380 = None
    mul_861: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_384);  sub_268 = unsqueeze_384 = None
    mul_862: "f32[128]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_33);  sum_154 = squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_295: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_862, [128, 1, 1, 1]);  mul_862 = None
    mul_863: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_295, 0.07450538873672485);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_296: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_861, [128, 64, 3, 3]);  mul_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_102: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(clone_8)
    full_38: "f32[8, 128, 28, 28]" = torch.ops.aten.full.default([8, 128, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_269: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_38, sigmoid_102);  full_38 = None
    mul_864: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(clone_8, sub_269);  clone_8 = sub_269 = None
    add_126: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_864, 1);  mul_864 = None
    mul_865: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_126);  sigmoid_102 = add_126 = None
    mul_866: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_294, mul_865);  getitem_294 = mul_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_155: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_866, [0, 2, 3])
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_866, mul_66, view_47, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_866 = mul_66 = view_47 = None
    getitem_297: "f32[8, 128, 28, 28]" = convolution_backward_61[0]
    getitem_298: "f32[128, 64, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_297: "f32[1, 128, 576]" = torch.ops.aten.view.default(getitem_298, [1, 128, 576]);  getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_385: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_386: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 2])
    sub_270: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386)
    mul_867: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_297, sub_270);  sub_270 = None
    sum_157: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2]);  mul_867 = None
    mul_868: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, 0.001736111111111111);  sum_156 = None
    unsqueeze_387: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_388: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    mul_869: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, 0.001736111111111111)
    mul_870: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_871: "f32[128]" = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    unsqueeze_389: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_390: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    mul_872: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, view_46);  view_46 = None
    unsqueeze_391: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_392: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    sub_271: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386);  view_45 = unsqueeze_386 = None
    mul_873: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_390);  sub_271 = unsqueeze_390 = None
    sub_272: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_297, mul_873);  view_297 = mul_873 = None
    sub_273: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_388);  sub_272 = unsqueeze_388 = None
    mul_874: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_392);  sub_273 = unsqueeze_392 = None
    mul_875: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_31);  sum_157 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_298: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_875, [128, 1, 1, 1]);  mul_875 = None
    mul_876: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_298, 0.07450538873672485);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_299: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_874, [128, 64, 3, 3]);  mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_103: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(clone_7)
    full_39: "f32[8, 128, 28, 28]" = torch.ops.aten.full.default([8, 128, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_274: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_39, sigmoid_103);  full_39 = None
    mul_877: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(clone_7, sub_274);  clone_7 = sub_274 = None
    add_127: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_877, 1);  mul_877 = None
    mul_878: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_127);  sigmoid_103 = add_127 = None
    mul_879: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_297, mul_878);  getitem_297 = mul_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_158: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3])
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_879, mul_62, view_44, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_879 = mul_62 = view_44 = None
    getitem_300: "f32[8, 512, 28, 28]" = convolution_backward_62[0]
    getitem_301: "f32[128, 512, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_300: "f32[1, 128, 512]" = torch.ops.aten.view.default(getitem_301, [1, 128, 512]);  getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_393: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_28, 0);  squeeze_28 = None
    unsqueeze_394: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    sum_159: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_300, [0, 2])
    sub_275: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394)
    mul_880: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_300, sub_275);  sub_275 = None
    sum_160: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2]);  mul_880 = None
    mul_881: "f32[128]" = torch.ops.aten.mul.Tensor(sum_159, 0.001953125);  sum_159 = None
    unsqueeze_395: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_396: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    mul_882: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, 0.001953125)
    mul_883: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, squeeze_29)
    mul_884: "f32[128]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_397: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_398: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    mul_885: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, view_43);  view_43 = None
    unsqueeze_399: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_400: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    sub_276: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394);  view_42 = unsqueeze_394 = None
    mul_886: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_398);  sub_276 = unsqueeze_398 = None
    sub_277: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_300, mul_886);  view_300 = mul_886 = None
    sub_278: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_396);  sub_277 = unsqueeze_396 = None
    mul_887: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_400);  sub_278 = unsqueeze_400 = None
    mul_888: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_29);  sum_160 = squeeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_301: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_888, [128, 1, 1, 1]);  mul_888 = None
    mul_889: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_301, 0.07902489841601695);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_302: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(mul_887, [128, 512, 1, 1]);  mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_890: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_300, 0.9805806756909201);  getitem_300 = None
    sigmoid_104: "f32[8, 512, 28, 28]" = torch.ops.aten.sigmoid.default(add_15)
    full_40: "f32[8, 512, 28, 28]" = torch.ops.aten.full.default([8, 512, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_279: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(full_40, sigmoid_104);  full_40 = None
    mul_891: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_15, sub_279);  add_15 = sub_279 = None
    add_128: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Scalar(mul_891, 1);  mul_891 = None
    mul_892: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_104, add_128);  sigmoid_104 = add_128 = None
    mul_893: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_890, mul_892);  mul_890 = mul_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_129: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_834, mul_893);  mul_834 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_894: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_129, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_895: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_894, 2.0);  mul_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_896: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_895, convolution_15);  convolution_15 = None
    mul_897: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_895, sigmoid_12);  mul_895 = sigmoid_12 = None
    sum_161: "f32[8, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2, 3], True);  mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_64: "f32[8, 512, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_280: "f32[8, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_64)
    mul_898: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(alias_64, sub_280);  alias_64 = sub_280 = None
    mul_899: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_161, mul_898);  sum_161 = mul_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_162: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_899, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_899, relu_1, primals_178, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_899 = primals_178 = None
    getitem_303: "f32[8, 128, 1, 1]" = convolution_backward_63[0]
    getitem_304: "f32[512, 128, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_66: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_67: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_10: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_303);  le_10 = scalar_tensor_10 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_163: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_10, mean_1, primals_176, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = mean_1 = primals_176 = None
    getitem_306: "f32[8, 512, 1, 1]" = convolution_backward_64[0]
    getitem_307: "f32[128, 512, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 512, 28, 28]" = torch.ops.aten.expand.default(getitem_306, [8, 512, 28, 28]);  getitem_306 = None
    div_11: "f32[8, 512, 28, 28]" = torch.ops.aten.div.Scalar(expand_11, 784);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_130: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_897, div_11);  mul_897 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_164: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 2, 3])
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_130, mul_54, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_130 = mul_54 = view_41 = None
    getitem_309: "f32[8, 128, 28, 28]" = convolution_backward_65[0]
    getitem_310: "f32[512, 128, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_303: "f32[1, 512, 128]" = torch.ops.aten.view.default(getitem_310, [1, 512, 128]);  getitem_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_401: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_26, 0);  squeeze_26 = None
    unsqueeze_402: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    sum_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2])
    sub_281: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402)
    mul_900: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_303, sub_281);  sub_281 = None
    sum_166: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 2]);  mul_900 = None
    mul_901: "f32[512]" = torch.ops.aten.mul.Tensor(sum_165, 0.0078125);  sum_165 = None
    unsqueeze_403: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_404: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    mul_902: "f32[512]" = torch.ops.aten.mul.Tensor(sum_166, 0.0078125)
    mul_903: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, squeeze_27)
    mul_904: "f32[512]" = torch.ops.aten.mul.Tensor(mul_902, mul_903);  mul_902 = mul_903 = None
    unsqueeze_405: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_406: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    mul_905: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, view_40);  view_40 = None
    unsqueeze_407: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
    unsqueeze_408: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    sub_282: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402);  view_39 = unsqueeze_402 = None
    mul_906: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_406);  sub_282 = unsqueeze_406 = None
    sub_283: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_303, mul_906);  view_303 = mul_906 = None
    sub_284: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_404);  sub_283 = unsqueeze_404 = None
    mul_907: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_408);  sub_284 = unsqueeze_408 = None
    mul_908: "f32[512]" = torch.ops.aten.mul.Tensor(sum_166, squeeze_27);  sum_166 = squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_304: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_908, [512, 1, 1, 1]);  mul_908 = None
    mul_909: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_304, 0.1580497968320339);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_305: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_907, [512, 128, 1, 1]);  mul_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_105: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_14)
    full_41: "f32[8, 128, 28, 28]" = torch.ops.aten.full.default([8, 128, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_285: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_41, sigmoid_105);  full_41 = None
    mul_910: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, sub_285);  convolution_14 = sub_285 = None
    add_131: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_910, 1);  mul_910 = None
    mul_911: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_105, add_131);  sigmoid_105 = add_131 = None
    mul_912: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_309, mul_911);  getitem_309 = mul_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_167: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_912, [0, 2, 3])
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_912, mul_50, view_38, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_912 = mul_50 = view_38 = None
    getitem_312: "f32[8, 128, 28, 28]" = convolution_backward_66[0]
    getitem_313: "f32[128, 64, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_306: "f32[1, 128, 576]" = torch.ops.aten.view.default(getitem_313, [1, 128, 576]);  getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_409: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_410: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    sum_168: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_306, [0, 2])
    sub_286: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410)
    mul_913: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_306, sub_286);  sub_286 = None
    sum_169: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2]);  mul_913 = None
    mul_914: "f32[128]" = torch.ops.aten.mul.Tensor(sum_168, 0.001736111111111111);  sum_168 = None
    unsqueeze_411: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_412: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    mul_915: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, 0.001736111111111111)
    mul_916: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_917: "f32[128]" = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
    unsqueeze_413: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_414: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    mul_918: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, view_37);  view_37 = None
    unsqueeze_415: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_416: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    sub_287: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410);  view_36 = unsqueeze_410 = None
    mul_919: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_414);  sub_287 = unsqueeze_414 = None
    sub_288: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_306, mul_919);  view_306 = mul_919 = None
    sub_289: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_412);  sub_288 = unsqueeze_412 = None
    mul_920: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_416);  sub_289 = unsqueeze_416 = None
    mul_921: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_307: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_921, [128, 1, 1, 1]);  mul_921 = None
    mul_922: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_307, 0.07450538873672485);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_308: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_920, [128, 64, 3, 3]);  mul_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_106: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(clone_6)
    full_42: "f32[8, 128, 28, 28]" = torch.ops.aten.full.default([8, 128, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_290: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_42, sigmoid_106);  full_42 = None
    mul_923: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(clone_6, sub_290);  clone_6 = sub_290 = None
    add_132: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_923, 1);  mul_923 = None
    mul_924: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_132);  sigmoid_106 = add_132 = None
    mul_925: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_312, mul_924);  getitem_312 = mul_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_170: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_925, [0, 2, 3])
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_925, mul_46, view_35, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_925 = mul_46 = view_35 = None
    getitem_315: "f32[8, 128, 56, 56]" = convolution_backward_67[0]
    getitem_316: "f32[128, 64, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_309: "f32[1, 128, 576]" = torch.ops.aten.view.default(getitem_316, [1, 128, 576]);  getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_417: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_22, 0);  squeeze_22 = None
    unsqueeze_418: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    sum_171: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_309, [0, 2])
    sub_291: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418)
    mul_926: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_309, sub_291);  sub_291 = None
    sum_172: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_926, [0, 2]);  mul_926 = None
    mul_927: "f32[128]" = torch.ops.aten.mul.Tensor(sum_171, 0.001736111111111111);  sum_171 = None
    unsqueeze_419: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_420: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    mul_928: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, 0.001736111111111111)
    mul_929: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, squeeze_23)
    mul_930: "f32[128]" = torch.ops.aten.mul.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
    unsqueeze_421: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_930, 0);  mul_930 = None
    unsqueeze_422: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    mul_931: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, view_34);  view_34 = None
    unsqueeze_423: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_424: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    sub_292: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418);  view_33 = unsqueeze_418 = None
    mul_932: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_422);  sub_292 = unsqueeze_422 = None
    sub_293: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_309, mul_932);  view_309 = mul_932 = None
    sub_294: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_420);  sub_293 = unsqueeze_420 = None
    mul_933: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_424);  sub_294 = unsqueeze_424 = None
    mul_934: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_23);  sum_172 = squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_310: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_934, [128, 1, 1, 1]);  mul_934 = None
    mul_935: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_310, 0.07450538873672485);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_311: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_933, [128, 64, 3, 3]);  mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_107: "f32[8, 128, 56, 56]" = torch.ops.aten.sigmoid.default(clone_5)
    full_43: "f32[8, 128, 56, 56]" = torch.ops.aten.full.default([8, 128, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_295: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(full_43, sigmoid_107);  full_43 = None
    mul_936: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(clone_5, sub_295);  clone_5 = sub_295 = None
    add_133: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Scalar(mul_936, 1);  mul_936 = None
    mul_937: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_133);  sigmoid_107 = add_133 = None
    mul_938: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_315, mul_937);  getitem_315 = mul_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_173: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_938, [0, 2, 3])
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_938, mul_39, view_32, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_938 = view_32 = None
    getitem_318: "f32[8, 256, 56, 56]" = convolution_backward_68[0]
    getitem_319: "f32[128, 256, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_312: "f32[1, 128, 256]" = torch.ops.aten.view.default(getitem_319, [1, 128, 256]);  getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_425: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_20, 0);  squeeze_20 = None
    unsqueeze_426: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    sum_174: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_312, [0, 2])
    sub_296: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426)
    mul_939: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(view_312, sub_296);  sub_296 = None
    sum_175: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_939, [0, 2]);  mul_939 = None
    mul_940: "f32[128]" = torch.ops.aten.mul.Tensor(sum_174, 0.00390625);  sum_174 = None
    unsqueeze_427: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    unsqueeze_428: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    mul_941: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, 0.00390625)
    mul_942: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, squeeze_21)
    mul_943: "f32[128]" = torch.ops.aten.mul.Tensor(mul_941, mul_942);  mul_941 = mul_942 = None
    unsqueeze_429: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_430: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    mul_944: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, view_31);  view_31 = None
    unsqueeze_431: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_432: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    sub_297: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426);  view_30 = unsqueeze_426 = None
    mul_945: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_430);  sub_297 = unsqueeze_430 = None
    sub_298: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_312, mul_945);  view_312 = mul_945 = None
    sub_299: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_428);  sub_298 = unsqueeze_428 = None
    mul_946: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_432);  sub_299 = unsqueeze_432 = None
    mul_947: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_21);  sum_175 = squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_313: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_947, [128, 1, 1, 1]);  mul_947 = None
    mul_948: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_313, 0.11175808310508728);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_314: "f32[128, 256, 1, 1]" = torch.ops.aten.view.default(mul_946, [128, 256, 1, 1]);  mul_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_176: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 2, 3])
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_129, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_129 = avg_pool2d = view_29 = None
    getitem_321: "f32[8, 256, 28, 28]" = convolution_backward_69[0]
    getitem_322: "f32[512, 256, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_315: "f32[1, 512, 256]" = torch.ops.aten.view.default(getitem_322, [1, 512, 256]);  getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_433: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_434: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_315, [0, 2])
    sub_300: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434)
    mul_949: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_315, sub_300);  sub_300 = None
    sum_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2]);  mul_949 = None
    mul_950: "f32[512]" = torch.ops.aten.mul.Tensor(sum_177, 0.00390625);  sum_177 = None
    unsqueeze_435: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    unsqueeze_436: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    mul_951: "f32[512]" = torch.ops.aten.mul.Tensor(sum_178, 0.00390625)
    mul_952: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_953: "f32[512]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
    unsqueeze_437: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    unsqueeze_438: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    mul_954: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, view_28);  view_28 = None
    unsqueeze_439: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_440: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    sub_301: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434);  view_27 = unsqueeze_434 = None
    mul_955: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_438);  sub_301 = unsqueeze_438 = None
    sub_302: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_315, mul_955);  view_315 = mul_955 = None
    sub_303: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_436);  sub_302 = unsqueeze_436 = None
    mul_956: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_440);  sub_303 = unsqueeze_440 = None
    mul_957: "f32[512]" = torch.ops.aten.mul.Tensor(sum_178, squeeze_19);  sum_178 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_316: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_957, [512, 1, 1, 1]);  mul_957 = None
    mul_958: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_316, 0.11175808310508728);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_317: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_956, [512, 256, 1, 1]);  mul_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_2: "f32[8, 256, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_321, mul_39, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_321 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_134: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_318, avg_pool2d_backward_2);  getitem_318 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_959: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_134, 0.9805806756909201);  add_134 = None
    sigmoid_108: "f32[8, 256, 56, 56]" = torch.ops.aten.sigmoid.default(add_9)
    full_44: "f32[8, 256, 56, 56]" = torch.ops.aten.full.default([8, 256, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_304: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(full_44, sigmoid_108);  full_44 = None
    mul_960: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sub_304);  add_9 = sub_304 = None
    add_135: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Scalar(mul_960, 1);  mul_960 = None
    mul_961: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_108, add_135);  sigmoid_108 = add_135 = None
    mul_962: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_959, mul_961);  mul_959 = mul_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_963: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_962, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_964: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_963, 2.0);  mul_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_965: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_964, convolution_8);  convolution_8 = None
    mul_966: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_964, sigmoid_7);  mul_964 = sigmoid_7 = None
    sum_179: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_965, [2, 3], True);  mul_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_68: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_305: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_68)
    mul_967: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(alias_68, sub_305);  alias_68 = sub_305 = None
    mul_968: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_179, mul_967);  sum_179 = mul_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_968, [0, 2, 3])
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_968, relu, primals_174, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_968 = primals_174 = None
    getitem_324: "f32[8, 64, 1, 1]" = convolution_backward_70[0]
    getitem_325: "f32[256, 64, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_70: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_71: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_11: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_324);  le_11 = scalar_tensor_11 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_181: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(where_11, mean, primals_172, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean = primals_172 = None
    getitem_327: "f32[8, 256, 1, 1]" = convolution_backward_71[0]
    getitem_328: "f32[64, 256, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 256, 56, 56]" = torch.ops.aten.expand.default(getitem_327, [8, 256, 56, 56]);  getitem_327 = None
    div_12: "f32[8, 256, 56, 56]" = torch.ops.aten.div.Scalar(expand_12, 3136);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_136: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_966, div_12);  mul_966 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 2, 3])
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(add_136, mul_31, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_136 = mul_31 = view_26 = None
    getitem_330: "f32[8, 64, 56, 56]" = convolution_backward_72[0]
    getitem_331: "f32[256, 64, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_318: "f32[1, 256, 64]" = torch.ops.aten.view.default(getitem_331, [1, 256, 64]);  getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_441: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_16, 0);  squeeze_16 = None
    unsqueeze_442: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    sum_183: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_318, [0, 2])
    sub_306: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442)
    mul_969: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(view_318, sub_306);  sub_306 = None
    sum_184: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_969, [0, 2]);  mul_969 = None
    mul_970: "f32[256]" = torch.ops.aten.mul.Tensor(sum_183, 0.015625);  sum_183 = None
    unsqueeze_443: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
    unsqueeze_444: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    mul_971: "f32[256]" = torch.ops.aten.mul.Tensor(sum_184, 0.015625)
    mul_972: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
    mul_973: "f32[256]" = torch.ops.aten.mul.Tensor(mul_971, mul_972);  mul_971 = mul_972 = None
    unsqueeze_445: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_446: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    mul_974: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, view_25);  view_25 = None
    unsqueeze_447: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_448: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    sub_307: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442);  view_24 = unsqueeze_442 = None
    mul_975: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_446);  sub_307 = unsqueeze_446 = None
    sub_308: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_318, mul_975);  view_318 = mul_975 = None
    sub_309: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_444);  sub_308 = unsqueeze_444 = None
    mul_976: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_448);  sub_309 = unsqueeze_448 = None
    mul_977: "f32[256]" = torch.ops.aten.mul.Tensor(sum_184, squeeze_17);  sum_184 = squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_319: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_977, [256, 1, 1, 1]);  mul_977 = None
    mul_978: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_319, 0.22351616621017456);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_320: "f32[256, 64, 1, 1]" = torch.ops.aten.view.default(mul_976, [256, 64, 1, 1]);  mul_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_109: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_7)
    full_45: "f32[8, 64, 56, 56]" = torch.ops.aten.full.default([8, 64, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_310: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(full_45, sigmoid_109);  full_45 = None
    mul_979: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, sub_310);  convolution_7 = sub_310 = None
    add_137: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Scalar(mul_979, 1);  mul_979 = None
    mul_980: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_109, add_137);  sigmoid_109 = add_137 = None
    mul_981: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_330, mul_980);  getitem_330 = mul_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_185: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_981, [0, 2, 3])
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_981, mul_27, view_23, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_981 = mul_27 = view_23 = None
    getitem_333: "f32[8, 64, 56, 56]" = convolution_backward_73[0]
    getitem_334: "f32[64, 64, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_321: "f32[1, 64, 576]" = torch.ops.aten.view.default(getitem_334, [1, 64, 576]);  getitem_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_449: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_14, 0);  squeeze_14 = None
    unsqueeze_450: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    sum_186: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_321, [0, 2])
    sub_311: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450)
    mul_982: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(view_321, sub_311);  sub_311 = None
    sum_187: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2]);  mul_982 = None
    mul_983: "f32[64]" = torch.ops.aten.mul.Tensor(sum_186, 0.001736111111111111);  sum_186 = None
    unsqueeze_451: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_452: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    mul_984: "f32[64]" = torch.ops.aten.mul.Tensor(sum_187, 0.001736111111111111)
    mul_985: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, squeeze_15)
    mul_986: "f32[64]" = torch.ops.aten.mul.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
    unsqueeze_453: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
    unsqueeze_454: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    mul_987: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, view_22);  view_22 = None
    unsqueeze_455: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_456: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    sub_312: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450);  view_21 = unsqueeze_450 = None
    mul_988: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_454);  sub_312 = unsqueeze_454 = None
    sub_313: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_321, mul_988);  view_321 = mul_988 = None
    sub_314: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_452);  sub_313 = unsqueeze_452 = None
    mul_989: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_456);  sub_314 = unsqueeze_456 = None
    mul_990: "f32[64]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_15);  sum_187 = squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_322: "f32[64, 1, 1, 1]" = torch.ops.aten.view.default(mul_990, [64, 1, 1, 1]);  mul_990 = None
    mul_991: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_322, 0.07450538873672485);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_323: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_989, [64, 64, 3, 3]);  mul_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_110: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(clone_4)
    full_46: "f32[8, 64, 56, 56]" = torch.ops.aten.full.default([8, 64, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_315: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(full_46, sigmoid_110);  full_46 = None
    mul_992: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(clone_4, sub_315);  clone_4 = sub_315 = None
    add_138: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Scalar(mul_992, 1);  mul_992 = None
    mul_993: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_138);  sigmoid_110 = add_138 = None
    mul_994: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_333, mul_993);  getitem_333 = mul_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_188: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_994, [0, 2, 3])
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_994, mul_23, view_20, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_994 = mul_23 = view_20 = None
    getitem_336: "f32[8, 64, 56, 56]" = convolution_backward_74[0]
    getitem_337: "f32[64, 64, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_324: "f32[1, 64, 576]" = torch.ops.aten.view.default(getitem_337, [1, 64, 576]);  getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_457: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_458: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    sum_189: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_324, [0, 2])
    sub_316: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458)
    mul_995: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(view_324, sub_316);  sub_316 = None
    sum_190: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_995, [0, 2]);  mul_995 = None
    mul_996: "f32[64]" = torch.ops.aten.mul.Tensor(sum_189, 0.001736111111111111);  sum_189 = None
    unsqueeze_459: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_460: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    mul_997: "f32[64]" = torch.ops.aten.mul.Tensor(sum_190, 0.001736111111111111)
    mul_998: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_999: "f32[64]" = torch.ops.aten.mul.Tensor(mul_997, mul_998);  mul_997 = mul_998 = None
    unsqueeze_461: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_462: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    mul_1000: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, view_19);  view_19 = None
    unsqueeze_463: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_464: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    sub_317: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458);  view_18 = unsqueeze_458 = None
    mul_1001: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_462);  sub_317 = unsqueeze_462 = None
    sub_318: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_324, mul_1001);  view_324 = mul_1001 = None
    sub_319: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_460);  sub_318 = unsqueeze_460 = None
    mul_1002: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_464);  sub_319 = unsqueeze_464 = None
    mul_1003: "f32[64]" = torch.ops.aten.mul.Tensor(sum_190, squeeze_13);  sum_190 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_325: "f32[64, 1, 1, 1]" = torch.ops.aten.view.default(mul_1003, [64, 1, 1, 1]);  mul_1003 = None
    mul_1004: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_325, 0.07450538873672485);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_326: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_1002, [64, 64, 3, 3]);  mul_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_111: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(clone_3)
    full_47: "f32[8, 64, 56, 56]" = torch.ops.aten.full.default([8, 64, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_320: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(full_47, sigmoid_111);  full_47 = None
    mul_1005: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(clone_3, sub_320);  clone_3 = sub_320 = None
    add_139: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Scalar(mul_1005, 1);  mul_1005 = None
    mul_1006: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_139);  sigmoid_111 = add_139 = None
    mul_1007: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_336, mul_1006);  getitem_336 = mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_191: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3])
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1007, mul_16, view_17, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1007 = view_17 = None
    getitem_339: "f32[8, 128, 56, 56]" = convolution_backward_75[0]
    getitem_340: "f32[64, 128, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_327: "f32[1, 64, 128]" = torch.ops.aten.view.default(getitem_340, [1, 64, 128]);  getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_465: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_10, 0);  squeeze_10 = None
    unsqueeze_466: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    sum_192: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_327, [0, 2])
    sub_321: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466)
    mul_1008: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(view_327, sub_321);  sub_321 = None
    sum_193: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1008, [0, 2]);  mul_1008 = None
    mul_1009: "f32[64]" = torch.ops.aten.mul.Tensor(sum_192, 0.0078125);  sum_192 = None
    unsqueeze_467: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_468: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    mul_1010: "f32[64]" = torch.ops.aten.mul.Tensor(sum_193, 0.0078125)
    mul_1011: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
    mul_1012: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1010, mul_1011);  mul_1010 = mul_1011 = None
    unsqueeze_469: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_470: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    mul_1013: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, view_16);  view_16 = None
    unsqueeze_471: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    unsqueeze_472: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    sub_322: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466);  view_15 = unsqueeze_466 = None
    mul_1014: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_470);  sub_322 = unsqueeze_470 = None
    sub_323: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_327, mul_1014);  view_327 = mul_1014 = None
    sub_324: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_468);  sub_323 = unsqueeze_468 = None
    mul_1015: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_472);  sub_324 = unsqueeze_472 = None
    mul_1016: "f32[64]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_11);  sum_193 = squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_328: "f32[64, 1, 1, 1]" = torch.ops.aten.view.default(mul_1016, [64, 1, 1, 1]);  mul_1016 = None
    mul_1017: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_328, 0.1580497968320339);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_329: "f32[64, 128, 1, 1]" = torch.ops.aten.view.default(mul_1015, [64, 128, 1, 1]);  mul_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_194: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 2, 3])
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_962, mul_16, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_962 = mul_16 = view_14 = None
    getitem_342: "f32[8, 128, 56, 56]" = convolution_backward_76[0]
    getitem_343: "f32[256, 128, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    add_140: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_330: "f32[1, 256, 128]" = torch.ops.aten.view.default(getitem_343, [1, 256, 128]);  getitem_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_473: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_8, 0);  squeeze_8 = None
    unsqueeze_474: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 2])
    sub_325: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474)
    mul_1018: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(view_330, sub_325);  sub_325 = None
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2]);  mul_1018 = None
    mul_1019: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, 0.0078125);  sum_195 = None
    unsqueeze_475: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_476: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    mul_1020: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, 0.0078125)
    mul_1021: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
    mul_1022: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
    unsqueeze_477: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    unsqueeze_478: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    mul_1023: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, view_13);  view_13 = None
    unsqueeze_479: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_480: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    sub_326: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474);  view_12 = unsqueeze_474 = None
    mul_1024: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_478);  sub_326 = unsqueeze_478 = None
    sub_327: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_330, mul_1024);  view_330 = mul_1024 = None
    sub_328: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_476);  sub_327 = unsqueeze_476 = None
    mul_1025: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_480);  sub_328 = unsqueeze_480 = None
    mul_1026: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, squeeze_9);  sum_196 = squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_331: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1026, [256, 1, 1, 1]);  mul_1026 = None
    mul_1027: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_331, 0.1580497968320339);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_332: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_1025, [256, 128, 1, 1]);  mul_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1028: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_140, 1.0);  add_140 = None
    sigmoid_112: "f32[8, 128, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_3)
    full_48: "f32[8, 128, 56, 56]" = torch.ops.aten.full.default([8, 128, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_329: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(full_48, sigmoid_112);  full_48 = None
    mul_1029: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, sub_329);  convolution_3 = sub_329 = None
    add_141: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Scalar(mul_1029, 1);  mul_1029 = None
    mul_1030: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_112, add_141);  sigmoid_112 = add_141 = None
    mul_1031: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1028, mul_1030);  mul_1028 = mul_1030 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1031, [0, 2, 3])
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1031, mul_11, view_11, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1031 = mul_11 = view_11 = None
    getitem_345: "f32[8, 64, 112, 112]" = convolution_backward_77[0]
    getitem_346: "f32[128, 64, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_333: "f32[1, 128, 576]" = torch.ops.aten.view.default(getitem_346, [1, 128, 576]);  getitem_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_481: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_482: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    sum_198: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 2])
    sub_330: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482)
    mul_1032: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_333, sub_330);  sub_330 = None
    sum_199: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 2]);  mul_1032 = None
    mul_1033: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, 0.001736111111111111);  sum_198 = None
    unsqueeze_483: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_484: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    mul_1034: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, 0.001736111111111111)
    mul_1035: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1036: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1034, mul_1035);  mul_1034 = mul_1035 = None
    unsqueeze_485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    mul_1037: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, view_10);  view_10 = None
    unsqueeze_487: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_488: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    sub_331: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482);  view_9 = unsqueeze_482 = None
    mul_1038: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_486);  sub_331 = unsqueeze_486 = None
    sub_332: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_333, mul_1038);  view_333 = mul_1038 = None
    sub_333: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_484);  sub_332 = unsqueeze_484 = None
    mul_1039: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_488);  sub_333 = unsqueeze_488 = None
    mul_1040: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_7);  sum_199 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_334: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1040, [128, 1, 1, 1]);  mul_1040 = None
    mul_1041: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_334, 0.07450538873672485);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_335: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_1039, [128, 64, 3, 3]);  mul_1039 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_113: "f32[8, 64, 112, 112]" = torch.ops.aten.sigmoid.default(clone_2)
    full_49: "f32[8, 64, 112, 112]" = torch.ops.aten.full.default([8, 64, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_334: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(full_49, sigmoid_113);  full_49 = None
    mul_1042: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(clone_2, sub_334);  clone_2 = sub_334 = None
    add_142: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Scalar(mul_1042, 1);  mul_1042 = None
    mul_1043: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_113, add_142);  sigmoid_113 = add_142 = None
    mul_1044: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_345, mul_1043);  getitem_345 = mul_1043 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_200: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1044, [0, 2, 3])
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1044, mul_7, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1044 = mul_7 = view_8 = None
    getitem_348: "f32[8, 32, 112, 112]" = convolution_backward_78[0]
    getitem_349: "f32[64, 32, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_336: "f32[1, 64, 288]" = torch.ops.aten.view.default(getitem_349, [1, 64, 288]);  getitem_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_489: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_4, 0);  squeeze_4 = None
    unsqueeze_490: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    sum_201: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 2])
    sub_335: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490)
    mul_1045: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(view_336, sub_335);  sub_335 = None
    sum_202: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2]);  mul_1045 = None
    mul_1046: "f32[64]" = torch.ops.aten.mul.Tensor(sum_201, 0.003472222222222222);  sum_201 = None
    unsqueeze_491: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_492: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    mul_1047: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, 0.003472222222222222)
    mul_1048: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
    mul_1049: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_493: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
    unsqueeze_494: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    mul_1050: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, view_7);  view_7 = None
    unsqueeze_495: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_496: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    sub_336: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490);  view_6 = unsqueeze_490 = None
    mul_1051: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_494);  sub_336 = unsqueeze_494 = None
    sub_337: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_336, mul_1051);  view_336 = mul_1051 = None
    sub_338: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_492);  sub_337 = unsqueeze_492 = None
    mul_1052: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_496);  sub_338 = unsqueeze_496 = None
    mul_1053: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_5);  sum_202 = squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_337: "f32[64, 1, 1, 1]" = torch.ops.aten.view.default(mul_1053, [64, 1, 1, 1]);  mul_1053 = None
    mul_1054: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_337, 0.10536653122135592);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_338: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_1052, [64, 32, 3, 3]);  mul_1052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_114: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(clone_1)
    full_50: "f32[8, 32, 112, 112]" = torch.ops.aten.full.default([8, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_339: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_50, sigmoid_114);  full_50 = None
    mul_1055: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(clone_1, sub_339);  clone_1 = sub_339 = None
    add_143: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_1055, 1);  mul_1055 = None
    mul_1056: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_114, add_143);  sigmoid_114 = add_143 = None
    mul_1057: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_348, mul_1056);  getitem_348 = mul_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_203: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3])
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1057, mul_3, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1057 = mul_3 = view_5 = None
    getitem_351: "f32[8, 16, 112, 112]" = convolution_backward_79[0]
    getitem_352: "f32[32, 16, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_339: "f32[1, 32, 144]" = torch.ops.aten.view.default(getitem_352, [1, 32, 144]);  getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_497: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
    unsqueeze_498: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    sum_204: "f32[32]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 2])
    sub_340: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498)
    mul_1058: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(view_339, sub_340);  sub_340 = None
    sum_205: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1058, [0, 2]);  mul_1058 = None
    mul_1059: "f32[32]" = torch.ops.aten.mul.Tensor(sum_204, 0.006944444444444444);  sum_204 = None
    unsqueeze_499: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
    unsqueeze_500: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    mul_1060: "f32[32]" = torch.ops.aten.mul.Tensor(sum_205, 0.006944444444444444)
    mul_1061: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
    mul_1062: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1060, mul_1061);  mul_1060 = mul_1061 = None
    unsqueeze_501: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_502: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    mul_1063: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, view_4);  view_4 = None
    unsqueeze_503: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_504: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    sub_341: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498);  view_3 = unsqueeze_498 = None
    mul_1064: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_502);  sub_341 = unsqueeze_502 = None
    sub_342: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_339, mul_1064);  view_339 = mul_1064 = None
    sub_343: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_500);  sub_342 = unsqueeze_500 = None
    mul_1065: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_504);  sub_343 = unsqueeze_504 = None
    mul_1066: "f32[32]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_3);  sum_205 = squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_340: "f32[32, 1, 1, 1]" = torch.ops.aten.view.default(mul_1066, [32, 1, 1, 1]);  mul_1066 = None
    mul_1067: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_340, 0.1490107774734497);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_341: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_1065, [32, 16, 3, 3]);  mul_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_115: "f32[8, 16, 112, 112]" = torch.ops.aten.sigmoid.default(clone)
    full_51: "f32[8, 16, 112, 112]" = torch.ops.aten.full.default([8, 16, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_344: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(full_51, sigmoid_115);  full_51 = None
    mul_1068: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(clone, sub_344);  clone = sub_344 = None
    add_144: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Scalar(mul_1068, 1);  mul_1068 = None
    mul_1069: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_115, add_144);  sigmoid_115 = add_144 = None
    mul_1070: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_351, mul_1069);  getitem_351 = mul_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_206: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3])
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1070, primals_222, view_2, [16], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1070 = primals_222 = view_2 = None
    getitem_355: "f32[16, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_342: "f32[1, 16, 27]" = torch.ops.aten.view.default(getitem_355, [1, 16, 27]);  getitem_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    unsqueeze_505: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_506: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    sum_207: "f32[16]" = torch.ops.aten.sum.dim_IntList(view_342, [0, 2])
    sub_345: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, unsqueeze_506)
    mul_1071: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(view_342, sub_345);  sub_345 = None
    sum_208: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1071, [0, 2]);  mul_1071 = None
    mul_1072: "f32[16]" = torch.ops.aten.mul.Tensor(sum_207, 0.037037037037037035);  sum_207 = None
    unsqueeze_507: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_508: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    mul_1073: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, 0.037037037037037035)
    mul_1074: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1075: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1073, mul_1074);  mul_1073 = mul_1074 = None
    unsqueeze_509: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_510: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    mul_1076: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, view_1);  view_1 = None
    unsqueeze_511: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
    unsqueeze_512: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    sub_346: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, unsqueeze_506);  view = unsqueeze_506 = None
    mul_1077: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_510);  sub_346 = unsqueeze_510 = None
    sub_347: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_342, mul_1077);  view_342 = mul_1077 = None
    sub_348: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_508);  sub_347 = unsqueeze_508 = None
    mul_1078: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_512);  sub_348 = unsqueeze_512 = None
    mul_1079: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, squeeze_1);  sum_208 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_343: "f32[16, 1, 1, 1]" = torch.ops.aten.view.default(mul_1079, [16, 1, 1, 1]);  mul_1079 = None
    mul_1080: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_343, 0.34412564994580647);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_344: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_1078, [16, 3, 3, 3]);  mul_1078 = None
    return pytree.tree_unflatten([addmm, view_344, mul_1080, sum_206, view_341, mul_1067, sum_203, view_338, mul_1054, sum_200, view_335, mul_1041, sum_197, view_332, mul_1027, sum_194, view_329, mul_1017, sum_191, view_326, mul_1004, sum_188, view_323, mul_991, sum_185, view_320, mul_978, sum_182, view_317, mul_958, sum_176, view_314, mul_948, sum_173, view_311, mul_935, sum_170, view_308, mul_922, sum_167, view_305, mul_909, sum_164, view_302, mul_889, sum_158, view_299, mul_876, sum_155, view_296, mul_863, sum_152, view_293, mul_850, sum_149, view_290, mul_830, sum_143, view_287, mul_820, sum_140, view_284, mul_807, sum_137, view_281, mul_794, sum_134, view_278, mul_781, sum_131, view_275, mul_761, sum_125, view_272, mul_748, sum_122, view_269, mul_735, sum_119, view_266, mul_722, sum_116, view_263, mul_702, sum_110, view_260, mul_689, sum_107, view_257, mul_676, sum_104, view_254, mul_663, sum_101, view_251, mul_643, sum_95, view_248, mul_630, sum_92, view_245, mul_617, sum_89, view_242, mul_604, sum_86, view_239, mul_584, sum_80, view_236, mul_571, sum_77, view_233, mul_558, sum_74, view_230, mul_545, sum_71, view_227, mul_525, sum_65, view_224, mul_512, sum_62, view_221, mul_499, sum_59, view_218, mul_486, sum_56, view_215, mul_466, sum_50, view_212, mul_456, sum_47, view_209, mul_443, sum_44, view_206, mul_430, sum_41, view_203, mul_417, sum_38, view_200, mul_397, sum_32, view_197, mul_384, sum_29, view_194, mul_371, sum_26, view_191, mul_358, sum_23, view_188, mul_338, sum_17, view_185, mul_325, sum_14, view_182, mul_312, sum_11, view_179, mul_299, sum_8, view_176, mul_283, sum_2, getitem_328, sum_181, getitem_325, sum_180, getitem_307, sum_163, getitem_304, sum_162, getitem_289, sum_148, getitem_286, sum_147, getitem_268, sum_130, getitem_265, sum_129, getitem_250, sum_115, getitem_247, sum_114, getitem_232, sum_100, getitem_229, sum_99, getitem_214, sum_85, getitem_211, sum_84, getitem_196, sum_70, getitem_193, sum_69, getitem_178, sum_55, getitem_175, sum_54, getitem_157, sum_37, getitem_154, sum_36, getitem_139, sum_22, getitem_136, sum_21, getitem_121, sum_7, getitem_118, sum_6, permute_4, view_172, None], self._out_spec)
    